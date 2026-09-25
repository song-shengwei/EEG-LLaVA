#!/usr/bin/env python3
"""Shared helpers for the A1 edge-deployment evaluation (M1-M5) of the final 55-token EEG-LLaVA.

Everything that determines a model output comes from copies of the code base that produced the
locked results (code/llamaG, code/H_dual_branch).  The copies differ from the originals only by
the two patches documented in code/PATCHES.md (autocast device routing; removal of a hard-coded
server path).  This module only

  1. resolves package-relative paths (the originals hard-code server paths);
  2. builds the model exactly like code/reference/profile_r1a.py::load_model and
     code/reference/r0_verify_checkpoint.py::build_and_load (strict state-dict load);
  3. sets the autocast around the LLM call to the device and precision under test;
  4. reproduces the per-segment loss-based score of
     code/reference/train_fold_clean.py::evaluate_loss_based;
  5. measures latency and memory: CUDA allocator peak exactly as on the RTX 4090, and on the
     CPU the kernel-tracked peak resident set size (VmHWM) of the process.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import pickle
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# [release] the edge-evaluation package layout was mapped onto the repository:
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT
CODE_DIR = REPO_ROOT / "src"
LLAMAG_DIR = CODE_DIR / "llamaG"
DUAL_DIR = CODE_DIR / "H_dual_branch"
QWEN_DIR = Path(os.environ.get("EEGLLAVA_LLM", REPO_ROOT / "pretrained" / "Qwen3-0.6B"))
CBRAMOD_INIT = Path(os.environ.get("EEGLLAVA_CBRAMOD", REPO_ROOT / "pretrained" / "cbramod" / "pretrained_weights.pth"))
LMDB_DIR = Path(os.environ.get("EEGLLAVA_LMDB", REPO_ROOT / "data" / "processed_lmdb"))
REFERENCE_DIR = REPO_ROOT / "deployment" / "edge" / "reference"
RESULTS_DIR = REPO_ROOT / "outputs" / "edge"

CHECKPOINTS = {
    # Protocol 2 (primary, subject-disjoint), R1a seed 1234, fold 0.  The RTX 4090 profile in
    # the paper (94.4 ms, 1,305 MiB) was measured on exactly this file (same SHA-256).
    "p2_r1a_fold0": Path(os.environ.get("EEGLLAVA_CKPT_DIR", REPO_ROOT / "checkpoints")) / "protocol2_R1a_seed1234" / "fold_0_best.pth",  # [release]
    # Protocol 1 (eye-disjoint), reported seed-42 checkpoint (Table 3: 76.6% BAcc, 0.829 AUC).
    "p1_seed42": Path(os.environ.get("EEGLLAVA_CKPT_DIR", REPO_ROOT / "checkpoints")) / "protocol1_seed42" / "fold_0_best.pth",  # [release]
}
REFERENCE_SCORES = {
    "p2_r1a_fold0": REFERENCE_DIR / "p2_r1a_fold0_test_rtx4090.json",
    "p1_seed42": REFERENCE_DIR / "p1_seed42_test_rtx4090.json",
}
SUBJECT_PREDICTIONS = REFERENCE_DIR / "p2_r1a_subject_predictions_rtx4090.csv"
PROFILE_SAMPLE_KEY = "P_0604_P5_l_OpenBCISession_2025-06-04_15-43-18_0"  # 4090 profile sample
PINNED = {
    "python": "3.12.13", "torch": "2.6.0", "transformers": "5.5.0", "tokenizers": "0.22.2",
    "numpy": "2.4.4", "lmdb": "2.2.0", "psutil": "7.2.2", "scikit-learn": "1.8.0",
    "safetensors": "0.7.0", "huggingface_hub": "1.9.0",
}

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(LLAMAG_DIR))
sys.path.insert(0, str(DUAL_DIR))
import model.glaucoma_llava as _glaucoma_llava  # noqa: E402
from data.glaucoma_llava_dataset import STAGE2_ANSWER, STAGE2_TEMPLATE  # noqa: E402
from dual_branch import AuxTransformer, DualBranchEEGLlava  # noqa: E402

for _name, _module in list(sys.modules.items()):
    if _name.split(".")[0] in {"model", "data", "dual_branch"}:
        _file = getattr(_module, "__file__", None)
        if _file and not Path(_file).resolve().is_relative_to(CODE_DIR):
            raise ImportError(f"{_name} was imported from outside the package: {_file}")

PROMPT = STAGE2_TEMPLATE + "\n"          # identical to evaluate_loss_based / profile_r1a
ANSWER_HEALTHY = STAGE2_ANSWER[0]        # "The EEG signal indicates a healthy visual system."
ANSWER_GLAUCOMA = STAGE2_ANSWER[1]       # "The EEG signal indicates glaucoma."


# --------------------------------------------------------------------------------------------
# Autocast around the LLM call (code/PATCHES.md, patch 1)
# --------------------------------------------------------------------------------------------
# The original glaucoma_llava.py hard-codes torch.amp.autocast('cuda', dtype=torch.bfloat16)
# around the Qwen3 call in forward() and generate(); the packaged copy reads LLM_AUTOCAST instead:
#   bf16 on cuda -> autocast('cuda', bf16)   (the original behaviour, RTX 4090 configuration)
#   bf16 on cpu  -> autocast('cpu',  bf16)   (EEG branches stay fp32 outside the context and the
#                                             LLM runs in bf16 inside it, as on the 4090)
#   fp32 / int8  -> autocast disabled        (genuinely fp32 activations)
def set_autocast_route(device_type: str, enabled: bool) -> None:
    _glaucoma_llava.LLM_AUTOCAST.update(device_type=device_type, dtype=torch.bfloat16,
                                        enabled=enabled)


# --------------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (pass --overwrite or --tag)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def default_output(kind: str, checkpoint: str | None, device: str, precision: str,
                   tag: str | None) -> Path:
    parts = [kind] + ([checkpoint] if checkpoint else []) + [device, precision]
    if tag:
        parts.append(tag)
    return RESULTS_DIR / ("_".join(parts) + ".json")


def summarize(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def physical_cores() -> int:
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def setup_threads(threads: int | None) -> int:
    count = threads or int(os.environ.get("EEG_EDGE_THREADS", "0")) or physical_cores()
    torch.set_num_threads(count)
    return torch.get_num_threads()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_native_bf16() -> bool:
    """bf16 tensor-core support (Ampere / compute capability 8.0 and newer)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def gpu_arch_supported() -> bool:
    """True if this torch build ships kernels runnable on the GPU (same major, minor <= device)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    for arch in torch.cuda.get_arch_list():
        match = re.fullmatch(r"sm_(\d+?)(\d)", arch)
        if match and int(match.group(1)) == major and int(match.group(2)) <= minor:
            return True
    return False


# --------------------------------------------------------------------------------------------
# Platform description (written into every result file)
# --------------------------------------------------------------------------------------------
def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_model_and_flags() -> tuple[str, set[str]]:
    text = _read("/proc/cpuinfo") or ""
    model = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    flags = re.search(r"^flags\s*:\s*(.+)$", text, re.M)
    return (model.group(1).strip() if model else platform.processor() or "unknown",
            set(flags.group(1).split()) if flags else set())


def _power_source() -> dict:
    info: dict = {"ac_online": None, "battery_percent": None, "battery_status": None}
    base = Path("/sys/class/power_supply")
    if not base.is_dir():
        return info
    for supply in sorted(base.iterdir()):
        kind = _read(str(supply / "type"))
        if kind == "Mains":
            online = _read(str(supply / "online"))
            if online is not None:
                info["ac_online"] = online == "1"
        elif kind == "Battery":
            capacity = _read(str(supply / "capacity"))
            info["battery_percent"] = int(capacity) if capacity and capacity.isdigit() else None
            info["battery_status"] = _read(str(supply / "status"))
    return info


def _nvidia_smi() -> dict | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in output:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "driver_version": parts[1],
                         "memory_total": parts[2],
                         "power_limit": parts[3] if len(parts) > 3 else None})
    return {"gpus": rows}


def platform_info(device: torch.device | None = None, probe_gpu: bool = False) -> dict:
    import sklearn
    import tokenizers
    import transformers

    cpu_model, flags = _cpu_model_and_flags()
    frequency = psutil.cpu_freq()
    os_release = _read("/etc/os-release") or ""
    pretty = re.search(r'^PRETTY_NAME="?(.*?)"?$', os_release, re.M)
    info = {
        "os": pretty.group(1) if pretty else platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_mhz": float(frequency.max) if frequency and frequency.max else None,
        "cpu_flags_relevant": sorted(flags & {"avx2", "fma", "avx512f", "avx512_bf16",
                                              "avx512_vnni", "avx_vnni", "amx_bf16",
                                              "amx_int8", "amx_tile"}),
        "cpu_native_bf16": bool(flags & {"avx512_bf16", "amx_bf16"}),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "ram_total_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "power": _power_source(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env_threads": {key: os.environ.get(key) for key in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "EEG_EDGE_THREADS",
                         "CUDA_VISIBLE_DEVICES")},
        "device": str(device) if device is not None else None,
    }
    if (device is not None and device.type == "cuda") or probe_gpu:
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_mib": round(properties.total_memory / 1024**2),
                "native_bf16": gpu_native_bf16(),
                "arch_supported_by_this_torch_build": gpu_arch_supported(),
                "torch_arch_list": torch.cuda.get_arch_list(),
                "cudnn": torch.backends.cudnn.version(),
            }
        info["nvidia_smi"] = _nvidia_smi()
    return info


# --------------------------------------------------------------------------------------------
# Memory measurement
# --------------------------------------------------------------------------------------------
def _status_mib(field: str) -> float | None:
    text = _read("/proc/self/status")
    if not text:
        return None
    match = re.search(rf"^{field}:\s+(\d+)\s+kB", text, re.M)
    return int(match.group(1)) / 1024.0 if match else None


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def release_free_heap() -> bool:
    """gc + glibc malloc_trim so RSS reflects live memory, not freed load-time buffers."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return True
    except (OSError, AttributeError):
        return False


class PeakMemory:
    """Peak memory over a code region.

    CUDA: torch.cuda.max_memory_allocated after reset_peak_memory_stats -- the metric of the
          RTX 4090 profile (1,305 MiB).
    CPU:  peak resident set size of the whole process.  On Linux the kernel's VmHWM counter is
          reset by writing "5" to /proc/self/clear_refs, which gives an exact, zero-overhead peak.
          If that is not permitted, a 5 ms psutil sampler thread is used instead.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.method = None
        self._sampler = None
        self._stop = threading.Event()
        self._sampled_peak = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            sync(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.method = "torch.cuda.max_memory_allocated"
            return self
        try:
            with open("/proc/self/clear_refs", "w") as handle:
                handle.write("5")
            if _status_mib("VmHWM") is None:
                raise OSError("VmHWM unavailable")
            self.method = "VmHWM (reset via /proc/self/clear_refs)"
        except OSError:
            self.method = "psutil RSS sampled every 5 ms"
            self._sampled_peak = rss_mib()
            self._stop.clear()
            self._sampler = threading.Thread(target=self._sample, daemon=True)
            self._sampler.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            self._sampled_peak = max(self._sampled_peak, rss_mib())
            time.sleep(0.005)

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            sync(self.device)
            self.peak_mib = torch.cuda.max_memory_allocated(self.device) / 1024**2
        elif self._sampler is not None:
            self._stop.set()
            self._sampler.join()
            self.peak_mib = max(self._sampled_peak, rss_mib())
        else:
            self.peak_mib = _status_mib("VmHWM")
        return False


def memory_snapshot(device: torch.device) -> dict:
    snapshot = {
        "process_rss_mib": rss_mib(),
        # VmHWM: peak RSS since process start or since the last PeakMemory reset
        "process_vmhwm_mib": _status_mib("VmHWM"),
    }
    if device.type == "cuda":
        snapshot["cuda_allocated_mib"] = torch.cuda.memory_allocated(device) / 1024**2
        snapshot["cuda_reserved_mib"] = torch.cuda.memory_reserved(device) / 1024**2
    return snapshot


# --------------------------------------------------------------------------------------------
# Model loading (mirrors profile_r1a.py::validate_checkpoint / load_model)
# --------------------------------------------------------------------------------------------
def validate_checkpoint(checkpoint_path: Path) -> tuple[dict, str]:
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    manifest_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")
    for required in (checkpoint_path, sidecar, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing: {required}")
    actual = sha256_file(checkpoint_path)
    expected = sidecar.read_text().split()[0]
    manifest = json.loads(manifest_path.read_text())
    if actual != expected or manifest.get("sha256") != actual:
        raise RuntimeError(f"checkpoint SHA-256 validation failed for {checkpoint_path}")
    if manifest.get("checkpoint_format") != "dual_branch_complete_v1":
        raise RuntimeError(f"unexpected checkpoint format: {manifest}")
    for flag in ("contains_spectral_state", "contains_aux_encoder_state",
                 "contains_eeg_encoder_state"):
        if not manifest.get(flag):
            raise RuntimeError(f"checkpoint manifest flag is false: {flag}")
    return manifest, actual


def apply_precision(model, precision: str, device: torch.device) -> dict:
    if precision == "bf16":
        if device.type == "cuda" and not gpu_native_bf16():
            raise RuntimeError("this GPU has no native bf16 (compute capability < 8.0); "
                               "use --precision fp32 for the GPU row")
        # Same dtype layout as the RTX 4090 run: EEG encoders/mappings fp32, Qwen3 weights bf16,
        # LLM computation under bf16 autocast.
        set_autocast_route(device.type, enabled=True)
        return {"precision": "bf16", "llm_weights": "bfloat16", "eeg_modules": "float32",
                "autocast": f"{device.type}/bfloat16 around the LLM call (as on the RTX 4090)"}
    if precision == "fp32":
        model.float()
        set_autocast_route(device.type, enabled=False)
        return {"precision": "fp32", "llm_weights": "float32 (exact upcast of the bf16 "
                "checkpoint weights)", "eeg_modules": "float32", "autocast": "disabled"}
    if precision == "int8":
        if device.type != "cpu":
            raise RuntimeError("int8 dynamic quantisation is CPU-only")
        model.float()
        set_autocast_route("cpu", enabled=False)
        engines = torch.backends.quantized.supported_engines
        for engine in ("x86", "fbgemm", "qnnpack"):
            if engine in engines:
                torch.backends.quantized.engine = engine
                break
        torch.ao.quantization.quantize_dynamic(
            model.llm, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        return {"precision": "int8", "llm_weights": "Linear layers dynamic int8 "
                f"({torch.backends.quantized.engine}); embeddings fp32",
                "eeg_modules": "float32", "autocast": "disabled",
                "note": "optional row; M5 agreement is mandatory whenever int8 is reported"}
    raise ValueError(f"unknown precision {precision}")


def load_model(checkpoint_name: str, device: torch.device, precision: str,
               verify_sha256: bool = True):
    checkpoint_path = CHECKPOINTS[checkpoint_name]
    if verify_sha256:
        manifest, checkpoint_sha = validate_checkpoint(checkpoint_path)
    else:
        manifest = json.loads(checkpoint_path.with_suffix(".pth.manifest.json").read_text())
        checkpoint_sha = manifest.get("sha256")
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    specification = checkpoint["model_spec"]
    if not specification.get("use_spectral"):
        raise RuntimeError("checkpoint does not contain the spectral branch")
    model = DualBranchEEGLlava(
        llm_path=str(QWEN_DIR),
        eeg_encoder_weights=str(CBRAMOD_INIT),  # original: .../pretrained_weights.pth
        freeze_eeg_encoder=True,
        freeze_llm=False,
        eeg_dim=int(specification["eeg_dim"]),
        num_channels=int(specification["num_channels"]),
        num_patches=int(specification["num_patches"]),
        aux_encoder=AuxTransformer(),
        aux_trainable=bool(specification.get("aux_trainable", False)),
        use_spectral=True,
        n_spectral_tokens=int(specification["n_spectral_tokens"]),
    )
    incompatibility = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatibility}")
    fold = checkpoint.get("fold")
    best_epoch = checkpoint.get("best_val_epoch")
    del checkpoint
    precision_info = apply_precision(model, precision, device)
    model = model.to(device).eval()
    sync(device)
    load_seconds = time.perf_counter() - started
    info = {
        "name": checkpoint_name,
        "path": str(checkpoint_path.relative_to(PKG_ROOT)),
        "sha256": checkpoint_sha,
        "manifest": manifest,
        "fold": fold,
        "best_val_epoch": best_epoch,
        "model_spec": {k: v for k, v in specification.items()
                       if k not in ("llm_path", "eeg_encoder_init")},
        "load_seconds": load_seconds,
        "precision": precision_info,
        "eeg_tokens": {"cbramod": 30, "aux_transformer": model.aux_encoder.n_tokens,
                       "spectral": model.spectral.n_tokens,
                       "total": 30 + model.aux_encoder.n_tokens + model.spectral.n_tokens},
    }
    if info["eeg_tokens"]["total"] != 55:
        raise RuntimeError(f"expected the 55-token model, got {info['eeg_tokens']}")
    return model, info


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
class SegmentStore:
    """Read-only access to the preprocessed LMDB (6 x 5 x 200 float32 per 5-s segment)."""

    def __init__(self, directory: Path = LMDB_DIR):
        self.env = lmdb.open(str(directory), readonly=True, lock=False, readahead=False,
                             meminit=False)

    def pair(self, key: str) -> dict:
        with self.env.begin(write=False) as transaction:
            raw = transaction.get(key.encode())
        if raw is None:
            raise KeyError(f"sample key missing from LMDB: {key}")
        return pickle.loads(raw)

    def eeg(self, key: str) -> torch.Tensor:
        """Identical to profile_r1a.load_real_sample and FoldDataset: sample / 100, float32."""
        sample = np.asarray(self.pair(key)["sample"], dtype=np.float32) / 100.0
        return torch.from_numpy(sample).reshape(1, 6, 5, 200)

    def label(self, key: str) -> int:
        return int(self.pair(key)["label"])


def load_reference(checkpoint_name: str) -> dict:
    return json.loads(REFERENCE_SCORES[checkpoint_name].read_text())


def subject_of(key: str) -> str:
    """Participant id, identical to r0_subject_evaluator.subject_of: first three '_' fields."""
    parts = key.split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot derive subject from key: {key}")
    return "_".join(parts[:3])


# --------------------------------------------------------------------------------------------
# Scoring (mirrors profile_r1a.candidate / loss_based_score and evaluate_loss_based)
# --------------------------------------------------------------------------------------------
def candidate(model, answer: str, device: torch.device):
    prompt_ids = model.tokenizer(PROMPT, return_tensors="pt",
                                 add_special_tokens=False)["input_ids"]
    encoded = model.tokenizer(PROMPT + answer, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return ids, mask, labels


def candidates(model, device: torch.device):
    return candidate(model, ANSWER_HEALTHY, device), candidate(model, ANSWER_GLAUCOMA, device)


def loss_based_two_passes(model, eeg, cands) -> None:
    """M1 unit of work: the two template forward passes (profile_r1a.loss_based_score)."""
    for ids, mask, target in cands:
        model(eeg, ids, mask, target)


def segment_score(model, eeg, cands) -> tuple[float, float, float]:
    """Glaucoma probability s_g = e^-Lg / (e^-Lh + e^-Lg), as in evaluate_loss_based."""
    healthy, glaucoma = cands
    loss_h = model(eeg, *healthy).loss.item()
    loss_g = model(eeg, *glaucoma).loss.item()
    probabilities = F.softmax(torch.tensor([-loss_h, -loss_g]), dim=0)
    return float(probabilities[1].item()), float(loss_h), float(loss_g)


class GenerationCounter:
    """Counts the tokens actually produced by model.llm.generate (generation stops at EOS)."""

    def __init__(self, llm):
        self._original = llm.generate
        self.last_new_tokens = None
        llm.generate = self

    def __call__(self, *args, **kwargs):
        output = self._original(*args, **kwargs)
        self.last_new_tokens = int(output.shape[-1])
        return output


def generated_decision(text: str) -> int:
    """Discrete mode of the paper: glaucoma if the response contains 'glaucoma'."""
    return 1 if "glaucoma" in text.lower() else 0


# --------------------------------------------------------------------------------------------
# Deterministic report fields (verbatim from code/reference/plot_fig13_20_sync.py)
# --------------------------------------------------------------------------------------------
CHANNELS = ["PO3", "POz", "PO4", "O1", "Oz", "O2"]


def segment_features(sample: np.ndarray) -> dict:
    signal = sample.astype(np.float32).reshape(6, -1) / 100.0
    frequencies = np.fft.rfftfreq(signal.shape[1], d=1.0 / 200.0)
    mask = (frequencies >= 8.0) & (frequencies <= 11.8)
    power = np.array([(np.abs(np.fft.rfft(channel)) ** 2)[mask].mean() for channel in signal])
    return _derived_fields(power)


def _derived_fields(power: np.ndarray) -> dict:
    relative = power / (power.max() + 1e-12) * 100.0
    left = power[[0, 3]].mean()
    right = power[[2, 5]].mean()
    return {
        "power": power.tolist(),
        "relative_percent": relative.tolist(),
        "max_channel": CHANNELS[int(power.argmax())],
        "min_channel": CHANNELS[int(power.argmin())],
        "cv_percent": float(power.std() / (power.mean() + 1e-12) * 100.0),
        "asymmetry_percent": float(abs(left - right) / (left + right + 1e-12) * 100.0),
    }


def participant_features(samples: list[np.ndarray]) -> dict:
    """Participant-level fields: per-channel SSVEP-band power averaged over all segments,
    then the same derived fields as the per-segment report (Fig. 19)."""
    powers = np.stack([np.asarray(segment_features(sample)["power"]) for sample in samples])
    return _derived_fields(powers.mean(axis=0))


def report_text(measured: dict, positive: int) -> str:
    """Same fixed template as plot_fig13_20_sync.report_text; the decision selects the verdict."""
    relatives = measured["relative_percent"]
    minimum = str(measured["min_channel"])
    minimum_index = CHANNELS.index(minimum)
    verdict = "Positive" if positive else "Negative"
    return (
        f"SSVEP-band power peaks at {measured['max_channel']}; {minimum} is "
        f"{relatives[minimum_index]:.0f}% of peak. Spatial heterogeneity is "
        f"CV={measured['cv_percent']:.0f}% and left-right asymmetry is "
        f"{measured['asymmetry_percent']:.0f}%. -> {verdict} screening result."
    )


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def binary_metrics(labels, scores, rule: str) -> dict:
    """rule '>' : segment level (evaluate_loss_based: scores > 0.5);
       rule '>=': participant level (r0_subject_evaluator: score >= 0.5)."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > 0.5) if rule == ">" else (scores >= 0.5)
    predictions = predictions.astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    auc = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None
    return {
        "threshold_rule": f"score {rule} 0.5",
        "n": int(labels.size),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "accuracy": (tp + tn) / labels.size,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
        "confusion_matrix_[[tn,fp],[fn,tp]]": [[tn, fp], [fn, tp]],
    }


def mean_sd(values) -> dict:
    values = list(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }
