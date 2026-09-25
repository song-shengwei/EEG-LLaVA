import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.eeg_encoder import EEGEncoder

# [edge-eval patch 1/2] 原代码在 forward()/generate() 里写死 torch.amp.autocast('cuda', dtype=torch.bfloat16)。
# 在 CPU 上该上下文不起作用（CPU bf16 的 generate 会因 fp32/bf16 混合输入报错），在 CUDA 上又会把
# 本意为 fp32 的运行变成 bf16。改为读取下面这个可配置项；默认值与原代码逐字等价（RTX 4090 实测配置），
# 由 code/edge/common.py 按被测设备与精度设置。详见 code/PATCHES.md。
LLM_AUTOCAST = {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}


def _llm_autocast():
    return torch.amp.autocast(**LLM_AUTOCAST)


def vfi_pairwise_rank_loss(pred, tgt, min_gap=0.02, gain=8.0):
    """成对 RankNet 排序损失 (VFI 严重度【排序】那条路的核心)。

    动机: 冻结/微调编码器 + 绝对值 MSE 都塌成「预测均值」(见 notes/vfi_regression_实现记录.md),
    因为右偏分布下常数=均值就是 MSE 极小点。排序损失换一个目标: 只要求把 batch 内的眼**按 VFI 排对序**。
    常数预测会让所有成对差为 0 → softplus(0)=ln2 恒为正 → 排序损失【无法】被塌成常数最小化,
    从而逼模型拉开预测去匹配严重度顺序。这与两条 MSE 路正交, 且临床上「谁更重」比绝对 VFI 更实用。

    pred, tgt: (B,) ∈ [0,1] (归一化 VFI)。只对目标差 > min_gap 的清晰有序对计损 (滤掉近似并列/同眼零差对)。
    对满足 tgt_i > tgt_j 的对, 希望 pred_i > pred_j: loss = softplus(-gain*(pred_i - pred_j))。
    gain 把 [0,1] 尺度的预测差放大到有意义的 logit 量级。返回标量; 无有效对时返回 0。
    """
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)      # (B,B) pred_i - pred_j
    dt = tgt.unsqueeze(1) - tgt.unsqueeze(0)        # (B,B) tgt_i - tgt_j
    mask = (dt > min_gap)                           # tgt_i 明显高于 tgt_j (i 视野更好); 单调保序, 与严重度方向无关
    n = mask.sum()
    if n == 0:
        return pred.new_zeros(())
    loss = nn.functional.softplus(-gain * dp)       # 想要 dp>0
    return (loss * mask).sum() / n.clamp(min=1)


class EEGProjector(nn.Module):
    """MLP projector: maps EEG features (dim=200) to LLM hidden_size (dim=1024)."""

    def __init__(self, eeg_dim=200, llm_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(eeg_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        return self.proj(x)


class VFIHead(nn.Module):
    """VFI(视野指数) 回归头 —— 从冻结编码器特征直接回归一个全局严重度标量。

    与 SectorHead 同构(全局池化 + MLP)，但只输出 **1 个标量**(VFI/100 ∈ [0,1])：
      vfi_pred:      (B,)                         每眼 VFI 的归一化预测(sigmoid, 回归金标准来自 HFA 报告)
      region_embeds: (B, n_tokens, llm_dim)       注入 LLM 的「VFI token」，让报告能口语化该数值。

    可靠预测取 vfi_pred(回归头, MSE 监督)；LLM 生成的数字只作可读报告，不作评测口径。
    ‼️ VFI 只存在于青光眼臂(见 README_VF_merge.md)，健康眼共线, 训练/评测须显式声明策略。
    """

    def __init__(self, eeg_dim=200, llm_dim=1024, n_tokens=3, hidden=256):
        super().__init__()
        self.n_tokens = n_tokens
        self.llm_dim = llm_dim
        self.trunk = nn.Sequential(
            nn.Linear(eeg_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.score_head = nn.Linear(hidden, 1)                 # -> 标量 logit -> sigmoid
        self.token_head = nn.Linear(hidden, n_tokens * llm_dim)

    def forward(self, feats):
        # feats: (B, num_tokens, eeg_dim) -> 池化 (B, eeg_dim)
        pooled = feats.mean(dim=1)
        h = self.trunk(pooled)                                 # (B, hidden)
        vfi_pred = torch.sigmoid(self.score_head(h)).squeeze(-1)  # (B,) ∈ [0,1]
        region_embeds = self.token_head(h).view(
            -1, self.n_tokens, self.llm_dim)                   # (B, n_tokens, llm_dim)
        return vfi_pred, region_embeds


class SectorHead(nn.Module):
    """视觉区域(逐环)检测头 —— 借鉴 nGoggle multifocal SSVEP 的分区域思想。

    输入: 编码器特征 (B, num_tokens, eeg_dim)；先跨 token 平均池化成每眼一个向量。
    输出:
      ring_scores: (B, n_ring)        每环【相对】响应强度 (回归, 由 CCA 派生目标监督)
      region_embeds: (B, n_ring_tokens, llm_dim)  注入 LLM 的「区域 token」，
                     让 LLM 在生成报告时能利用逐环空间信息。

    ⚠️ ring_scores 表征环间【相对】强弱 (见 build_ring_targets.py caveat)，非视野缺损金标准。
    """

    def __init__(self, eeg_dim=200, llm_dim=1024, n_ring=3, n_ring_tokens=3, hidden=256):
        super().__init__()
        self.n_ring = n_ring
        self.n_ring_tokens = n_ring_tokens
        self.llm_dim = llm_dim
        self.trunk = nn.Sequential(
            nn.Linear(eeg_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        # 回归出 n_ring 个相对强度分数 (供 aux loss)
        self.score_head = nn.Linear(hidden, n_ring)
        # 生成区域 token (注入 LLM)；展平到 n_ring_tokens * llm_dim 再 reshape
        self.token_head = nn.Linear(hidden, n_ring_tokens * llm_dim)

    def forward(self, feats):
        # feats: (B, num_tokens, eeg_dim) -> 池化 (B, eeg_dim)
        pooled = feats.mean(dim=1)
        h = self.trunk(pooled)                                  # (B, hidden)
        ring_scores = self.score_head(h)                        # (B, n_ring)
        region_embeds = self.token_head(h).view(
            -1, self.n_ring_tokens, self.llm_dim)               # (B, n_ring_tokens, llm_dim)
        return ring_scores, region_embeds


class RingQueryHead(nn.Module):
    """视觉区域(逐环)检测头 — 方法2：空间查询交叉注意力 (DETR/Perceiver 风格)。

    与 SectorHead(方法1, 全局平均池化 + MLP) 的本质区别：
      方法1 把 30 个 channel×patch 编码器 token **平均池化成 1 个向量**，再回归 3 环——
        所有环看到的是同一个被压扁的向量，无法做空间选择。
      方法2 保留 30 个 token，用 **n_ring 个可学习的「环查询向量」对 30 个 token 做
        多头交叉注意力**：每个环查询用自己学到的注意力权重去聚合 token，
        于是不同环可聚焦不同 channel/patch 子集 (如周边环偏向外侧枕区通道)，
        得到真正的空间选择性；注意力权重本身也是可解释产物 (每个环依赖哪些通道)。

    这正是 notes/视野分区检测_架构评估.md #3 推荐的「可学习空间查询向量 (DETR/Perceiver)」做法。

    输入: 编码器特征 (B, num_tokens=30, eeg_dim)。
    输出 (与 SectorHead 完全一致的接口, drop-in 可替换):
      ring_scores:   (B, n_ring)                    每环【相对】响应强度 (回归, 同一 CCA 目标监督)
      region_embeds: (B, n_ring_tokens, llm_dim)    注入 LLM 的区域 token

    ⚠️ 同样仅表征环间【相对】强弱 (见 build_ring_targets.py caveat)，非视野缺损金标准。
    """

    def __init__(self, eeg_dim=200, llm_dim=1024, n_ring=3, n_ring_tokens=3,
                 hidden=256, nhead=4):
        super().__init__()
        self.n_ring = n_ring
        self.n_ring_tokens = n_ring_tokens
        self.llm_dim = llm_dim
        # 输入投影: 把 30 个编码器 token 投到注意力维度，作为 key/value
        self.in_proj = nn.Sequential(
            nn.Linear(eeg_dim, hidden),
            nn.LayerNorm(hidden),
        )
        # n_ring 个可学习「环查询向量」(每环一个 query token)
        self.ring_queries = nn.Parameter(torch.randn(n_ring, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        # 每环上下文向量 -> 1 个相对强度分数 (供 aux MSE, 与方法1 同口径)
        self.score_head = nn.Linear(hidden, 1)
        # 每环上下文向量 -> 1 个区域 token (注入 LLM)；n_ring_tokens 应 == n_ring
        self.token_head = nn.Linear(hidden, llm_dim)
        # 调试/可解释用：保存最近一次注意力权重 (B, n_ring, num_tokens)
        self.last_attn = None

    def forward(self, feats):
        # feats: (B, num_tokens, eeg_dim)
        bz = feats.shape[0]
        kv = self.in_proj(feats)                                # (B, T, hidden)
        q = self.ring_queries.unsqueeze(0).expand(bz, -1, -1)   # (B, n_ring, hidden)
        ctx, attn_w = self.attn(q, kv, kv, need_weights=True)   # (B, n_ring, hidden)
        ctx = self.norm(ctx + q)                                # 残差
        ctx = ctx + self.ffn(ctx)
        self.last_attn = attn_w.detach()                        # (B, n_ring, T)
        ring_scores = self.score_head(ctx).squeeze(-1)          # (B, n_ring)
        # n_ring_tokens 个区域 token：取前 n_ring_tokens 个环查询的上下文 (默认全取)
        region_ctx = ctx[:, :self.n_ring_tokens, :]             # (B, n_ring_tokens, hidden)
        region_embeds = self.token_head(region_ctx)             # (B, n_ring_tokens, llm_dim)
        return ring_scores, region_embeds


class RetinoQueryHead(nn.Module):
    """视觉区域(20 区)检测头 — 方法3：分层视网膜拓扑查询头 (可能模型.md 模块2)。

    在 RingQueryHead(方法2, 纯 cross-attn) 基础上加两件 SleepLM/HEARTS 背书的东西：
      1) **电极身份 embedding**：30 个 token = 6 电极 × 5 patch；按所属电极给每个 token 加
         一个可学习码 (6 个)，让 20 个 region-query 能区分 PO3(左)/PO4(右)、O 排(下)/PO 排(上)——
         这是 SleepLM「通道轴可学习编码」的轻量版，给小样本一点视网膜拓扑结构。
      2) **可学习 query→电极注意力偏置** (n_ring × n_elec，**初始化 0 ⇒ 起步是 no-op**)：
         HEARTS 建议「注入临床先验」，但 6 干电极只有泛名 EXG0..5、硬编一张可能出错的
         视网膜↔电极图反而有害；故留一个**可学习**的偏置槽 (每个 region-query 对每个电极一个偏置)，
         让模型自己学 query 该偏向哪些电极，等价于「学出来的拓扑先验」，零先验风险。

    输出接口与 SectorHead / RingQueryHead **完全一致** (ring_scores, region_embeds)，drop-in 可替换：
      ring_scores:   (B, n_ring)                  20 区 logit (方法3 当 bottom-up BCE 的基底分数)
      region_embeds: (B, n_ring_tokens, llm_dim)  注入 LLM 的区域 token
    last_attn 保留最近一次注意力权重 (B, n_ring, 30) 供可解释。

    ⚠️ 同 E18/E19 红线：20 区只表征「相对偏弱」排序，**禁止断言「视野缺损」**。
    """

    def __init__(self, eeg_dim=200, llm_dim=1024, n_ring=20, n_ring_tokens=20,
                 hidden=256, nhead=4, num_channels=6, num_patches=5):
        super().__init__()
        self.n_ring = n_ring
        self.n_ring_tokens = n_ring_tokens
        self.llm_dim = llm_dim
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.in_proj = nn.Sequential(
            nn.Linear(eeg_dim, hidden),
            nn.LayerNorm(hidden),
        )
        # 6 个电极身份码 (加到对应电极的 5 个 patch token 上)
        self.elec_embed = nn.Parameter(torch.randn(num_channels, hidden) * 0.02)
        # n_ring 个可学习「区域查询向量」
        self.ring_queries = nn.Parameter(torch.randn(n_ring, hidden) * 0.02)
        # 可学习 query→电极偏置 (0 初始化, 起步 no-op; 训练中学出拓扑先验)
        self.query_elec_bias = nn.Parameter(torch.zeros(n_ring, num_channels))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.score_head = nn.Linear(hidden, 1)
        self.token_head = nn.Linear(hidden, llm_dim)
        self.last_attn = None
        # token t -> 电极 t // num_patches (随模型 .to(device) 移动)
        self.register_buffer(
            'elec_idx',
            torch.arange(num_channels).repeat_interleave(num_patches), persistent=False)

    def forward(self, feats):
        # feats: (B, num_tokens=30, eeg_dim)
        bz, T, _ = feats.shape
        kv = self.in_proj(feats)                                # (B, T, hidden)
        if T == self.elec_idx.shape[0]:                         # 加电极身份码 (T==30 时)
            kv = kv + self.elec_embed[self.elec_idx].unsqueeze(0)
        q = self.ring_queries.unsqueeze(0).expand(bz, -1, -1)   # (B, n_ring, hidden)
        # query->token 加性偏置 (n_ring, T)：电极偏置按 patch 展开；broadcast 到 batch/head
        attn_bias = None
        if T == self.elec_idx.shape[0]:
            attn_bias = self.query_elec_bias.repeat_interleave(
                self.num_patches, dim=1)                        # (n_ring, T)
        ctx, attn_w = self.attn(q, kv, kv, need_weights=True, attn_mask=attn_bias)
        ctx = self.norm(ctx + q)                                # 残差
        ctx = ctx + self.ffn(ctx)
        self.last_attn = attn_w.detach()                        # (B, n_ring, T)
        ring_scores = self.score_head(ctx).squeeze(-1)          # (B, n_ring)
        region_embeds = self.token_head(ctx[:, :self.n_ring_tokens, :])
        return ring_scores, region_embeds


def build_region_head(method, eeg_dim, llm_dim, n_ring, n_ring_tokens,
                      num_channels=6, num_patches=5):
    """按 region_method 选择区域检测头：
      'pool'=方法1(SectorHead 全局池化), 'query'=方法2(RingQueryHead 交叉注意力),
      'retino'=方法3(RetinoQueryHead 视网膜拓扑查询头, 带电极身份 + 可学习先验偏置)。"""
    if method == 'retino':
        return RetinoQueryHead(eeg_dim=eeg_dim, llm_dim=llm_dim,
                               n_ring=n_ring, n_ring_tokens=n_ring_tokens,
                               num_channels=num_channels, num_patches=num_patches)
    if method == 'query':
        return RingQueryHead(eeg_dim=eeg_dim, llm_dim=llm_dim,
                             n_ring=n_ring, n_ring_tokens=n_ring_tokens)
    return SectorHead(eeg_dim=eeg_dim, llm_dim=llm_dim,
                      n_ring=n_ring, n_ring_tokens=n_ring_tokens)


class EEGLlavaModel(nn.Module):
    """
    LLaVA-style model for EEG:
      EEG signal -> EEG encoder -> projector -> [EEG tokens]
      [EEG tokens] + [text tokens] -> Qwen3 LLM -> text output
    """

    def __init__(
        self,
        llm_path,
        eeg_encoder_weights=None,
        freeze_eeg_encoder=True,
        freeze_llm=False,
        eeg_dim=200,
        num_channels=6,
        num_patches=5,
        use_region=False,
        n_ring=3,
        n_ring_tokens=3,
        aux_weight=0.5,
        region_method='pool',
        use_vfi=False,
        n_vfi_tokens=3,
        vfi_aux_weight=1.0,
        vfi_loss='mse',
        vfi_rank_weight=1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.eeg_dim = eeg_dim
        self.num_eeg_tokens = num_channels * num_patches  # 6*5=30
        # --- 视觉区域(逐环)检测头配置 ---
        self.use_region = use_region
        self.n_ring = n_ring
        self.n_ring_tokens = n_ring_tokens if use_region else 0
        self.aux_weight = aux_weight
        # 区域头方法: 'pool'=方法1(SectorHead 全局池化), 'query'=方法2(RingQueryHead 交叉注意力)
        self.region_method = region_method
        # --- VFI(视野指数) 回归头配置 ---
        self.use_vfi = use_vfi
        self.n_vfi_tokens = n_vfi_tokens if use_vfi else 0
        self.vfi_aux_weight = vfi_aux_weight
        # VFI 辅助损失类型: 'mse'(旧, 绝对值回归) / 'huber'(鲁棒回归) /
        #   'rank'(纯成对排序) / 'rankhuber'(Huber 校准 + 排序正则, 推荐)。
        self.vfi_loss = vfi_loss
        self.vfi_rank_weight = vfi_rank_weight

        # 1. EEG Encoder
        self.eeg_encoder = EEGEncoder(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8
        )
        self.eeg_encoder.proj_out = nn.Identity()

        if eeg_encoder_weights is not None:
            state_dict = torch.load(eeg_encoder_weights, map_location='cpu')
            # Fine-tuned model has "backbone." prefix, remove it
            backbone_state = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    backbone_state[k[len('backbone.'):]] = v
            if backbone_state:
                self.eeg_encoder.load_state_dict(backbone_state, strict=False)
                print(f"Loaded EEG encoder weights from {eeg_encoder_weights}")
            else:
                # Try loading directly (pretrained weights without prefix)
                self.eeg_encoder.load_state_dict(state_dict, strict=False)
                print(f"Loaded EEG encoder weights (direct) from {eeg_encoder_weights}")

        if freeze_eeg_encoder:
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
            print("EEG encoder frozen")
        # encode_eeg 是否放开编码器梯度: 默认跟随 freeze_eeg_encoder;
        # 端到端微调(VFI 严重度)时由 unfreeze_encoder_last_n() 置 True(见其 docstring)。
        self.eeg_encoder_trainable = not freeze_eeg_encoder

        # 2. LLM (Qwen3)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True
        )
        llm_dim = self.llm.config.hidden_size  # 1024

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("LLM frozen")

        # 3. Projector (EEG dim -> LLM dim)
        self.projector = EEGProjector(eeg_dim=eeg_dim, llm_dim=llm_dim)

        # 3b. 视觉区域(逐环)检测头 —— 产出逐环相对强度 + 区域 token
        #     方法1='pool'(SectorHead) / 方法2='query'(RingQueryHead)，接口一致, 可对照消融。
        if self.use_region:
            self.sector_head = build_region_head(
                region_method, eeg_dim, llm_dim, n_ring, n_ring_tokens,
                num_channels=num_channels, num_patches=num_patches)
            print(f"[Region] head method = {region_method} "
                  f"({type(self.sector_head).__name__})")
        else:
            self.sector_head = None

        # 3c. VFI 回归头 —— 从冻结编码器特征回归全局视野指数(标量) + 注入 VFI token
        if self.use_vfi:
            self.vfi_head = VFIHead(eeg_dim=eeg_dim, llm_dim=llm_dim,
                                    n_tokens=n_vfi_tokens)
            print(f"[VFI] regression head enabled: n_vfi_tokens={n_vfi_tokens}, "
                  f"vfi_aux_weight={vfi_aux_weight}")
        else:
            self.vfi_head = None

        # 4. Special token for EEG placeholder
        self.eeg_token = "<eeg>"
        self.tokenizer.add_tokens([self.eeg_token], special_tokens=True)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.eeg_token_id = self.tokenizer.convert_tokens_to_ids(self.eeg_token)

    def unfreeze_encoder_last_n(self, n):
        """解冻 EEG 编码器最后 n 个 TransformerEncoderLayer, 供 VFI 严重度【端到端微调】:
        让 VFI MSE(及 LM)损失梯度回流进编码器, 检验冻结特征里线性不可解码的严重度信号
        能否被学出来 (见 notes/vfi_regression_实现记录.md 「冻结编码器塌成均值」)。
        n=0 保持全冻(等价旧行为); n=12 全解冻。返回是否存在可训练编码器层。"""
        layers = self.eeg_encoder.encoder.layers   # ModuleList, 12 层
        total = len(layers)
        n = max(0, min(int(n), total))
        for p in self.eeg_encoder.parameters():
            p.requires_grad = False
        for i in range(total - n, total):
            for p in layers[i].parameters():
                p.requires_grad = True
        self.eeg_encoder_trainable = n > 0
        n_tr = sum(p.numel() for p in self.eeg_encoder.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.eeg_encoder.parameters())
        print(f"[EncoderFT] 解冻编码器最后 {n}/{total} 层 "
              f"({n_tr:,}/{n_all:,} 参数, {100 * n_tr / max(n_all, 1):.1f}%), "
              f"trainable={self.eeg_encoder_trainable}")
        return self.eeg_encoder_trainable

    def encode_eeg(self, eeg_signal):
        """
        Args:
            eeg_signal: (batch, channels, patches, patch_size) = (B, 6, 5, 200)
        Returns:
            tokens: (batch, num_eeg_tokens [+ n_ring_tokens][+ n_vfi_tokens], llm_dim)
                    30 个全局 EEG token；use_region/use_vfi 时再拼接对应的头 token。
            ring_scores: (batch, n_ring) 或 None —— 逐环相对强度 (供 region aux loss)。
            vfi_pred:    (batch,) 或 None —— 归一化 VFI 预测 ∈ [0,1] (供 vfi aux loss)。
        """
        # EEG encoder: float32 precision.
        #   冻结(默认): no_grad, 省显存, 行为与旧版逐字节一致。
        #   端到端微调(unfreeze_encoder_last_n>0): 放开梯度, 让 VFI/LM 损失回流进编码器。
        if self.eeg_encoder_trainable:
            feats = self.eeg_encoder(eeg_signal)  # (B, 6, 5, 200), 保留计算图
        else:
            with torch.no_grad():
                feats = self.eeg_encoder(eeg_signal)  # (B, 6, 5, 200)
        bz = feats.shape[0]
        feats = feats.view(bz, -1, self.eeg_dim)  # (B, 30, 200)

        # Projector: float32 precision (trainable, keeps full precision for gradients)
        eeg_embeds = self.projector(feats)  # (B, 30, 1024), float32

        ring_scores = None
        if self.use_region:
            # 区域 token 与逐环分数都来自冻结编码器特征 feats (无 grad)，
            # 经可训练 sector_head 产出 -> 区域 token 拼到 EEG token 之后。
            ring_scores, region_embeds = self.sector_head(feats)  # (B,n_ring),(B,n_ring_tokens,1024)
            eeg_embeds = torch.cat([eeg_embeds, region_embeds.to(eeg_embeds.dtype)], dim=1)

        vfi_pred = None
        if self.use_vfi:
            # VFI 标量预测 + VFI token(拼到 EEG token 之后), 同样来自冻结编码器特征。
            vfi_pred, vfi_embeds = self.vfi_head(feats)  # (B,), (B,n_vfi_tokens,1024)
            eeg_embeds = torch.cat([eeg_embeds, vfi_embeds.to(eeg_embeds.dtype)], dim=1)

        return eeg_embeds, ring_scores, vfi_pred

    def forward(self, eeg_signal, input_ids, attention_mask, labels=None,
                sample_weights=None, ring_target=None, vfi_target=None,
                eeg_embeds=None):
        """
        Args:
            eeg_signal: (B, 6, 5, 200)；若传了 eeg_embeds 则可为 None
            input_ids: (B, seq_len) - text tokens with <eeg> placeholder tokens
            attention_mask: (B, seq_len)
            labels: (B, seq_len) - for computing loss, -100 for ignored positions
            ring_target: (B, n_ring) - 逐环相对强度回归目标 (use_region 时用于 aux loss)
            vfi_target:  (B,) - 归一化 VFI 回归目标 ∈ [0,1] (use_vfi 时用于 aux loss)
            eeg_embeds: (B 或 1, n_eeg_tokens, llm_dim) 预计算的 EEG embedding, 可选。
                        评估时同一段 EEG 要配多条候选序列(如 20 区 × 2 答案 = 40 条)，
                        传 (1, n, d) 即可让 encoder+projector **只算一次**再广播到整个 batch，
                        省掉 B 份重复编码 (E36 审查记录 P2-5)。给了它就不再调 encode_eeg,
                        故 ring_scores / vfi_pred 为 None, aux loss 自动跳过 —— 仅用于推理。
        Returns:
            HF output; outputs.loss = LM loss (+ aux_weight * ring MSE / vfi_aux_weight * VFI MSE);
            outputs.lm_loss / outputs.aux_loss 便于日志拆分。
        """
        # 1. Get EEG embeddings (+ 区域/VFI token) 和逐环分数/VFI 预测
        if eeg_embeds is None:
            eeg_embeds, ring_scores, vfi_pred = self.encode_eeg(eeg_signal)
        else:
            ring_scores = vfi_pred = None
            if eeg_embeds.shape[0] == 1 and input_ids.shape[0] > 1:
                eeg_embeds = eeg_embeds.expand(input_ids.shape[0], -1, -1)
            elif eeg_embeds.shape[0] != input_ids.shape[0]:
                raise ValueError(f'eeg_embeds batch {eeg_embeds.shape[0]} 与 input_ids batch '
                                 f'{input_ids.shape[0]} 不匹配(只允许相等或为 1)')

        # 2. Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, 1024)

        # 3. Replace <eeg> token positions with EEG embeddings
        batch_size = input_ids.shape[0]
        new_embeds = text_embeds.clone()
        new_attention_mask = attention_mask.clone()
        new_labels = labels.clone() if labels is not None else None

        # Build merged sequence: replace each <eeg> token with 95 EEG tokens
        merged_embeds_list = []
        merged_attention_list = []
        merged_labels_list = []

        for i in range(batch_size):
            eeg_positions = (input_ids[i] == self.eeg_token_id).nonzero(as_tuple=True)[0]

            if len(eeg_positions) == 0:
                merged_embeds_list.append(text_embeds[i])
                merged_attention_list.append(attention_mask[i])
                if labels is not None:
                    merged_labels_list.append(labels[i])
                continue

            # Split at the first <eeg> token position
            pos = eeg_positions[0].item()
            before = text_embeds[i, :pos]             # tokens before <eeg>
            after = text_embeds[i, pos + 1:]          # tokens after <eeg>
            eeg_emb = eeg_embeds[i]                   # (30[+n_ring_tokens], 1024)
            n_eeg = eeg_emb.shape[0]                  # 动态: 含区域 token, 勿写死

            merged_embed = torch.cat([before, eeg_emb, after], dim=0)
            merged_embeds_list.append(merged_embed)

            # Attention mask
            before_attn = attention_mask[i, :pos]
            after_attn = attention_mask[i, pos + 1:]
            eeg_attn = torch.ones(n_eeg, device=attention_mask.device,
                                  dtype=attention_mask.dtype)
            merged_attention_list.append(torch.cat([before_attn, eeg_attn, after_attn], dim=0))

            # Labels: EEG/region token positions should be ignored (-100)
            if labels is not None:
                before_lbl = labels[i, :pos]
                after_lbl = labels[i, pos + 1:]
                eeg_lbl = torch.full((n_eeg,), -100,
                                     device=labels.device, dtype=labels.dtype)
                merged_labels_list.append(torch.cat([before_lbl, eeg_lbl, after_lbl], dim=0))

        # Pad to same length
        max_len = max(e.shape[0] for e in merged_embeds_list)
        llm_dim = text_embeds.shape[-1]

        padded_embeds = torch.zeros(batch_size, max_len, llm_dim,
                                    device=text_embeds.device, dtype=text_embeds.dtype)
        padded_attention = torch.zeros(batch_size, max_len,
                                       device=attention_mask.device, dtype=attention_mask.dtype)
        padded_labels = torch.full((batch_size, max_len), -100,
                                    device=input_ids.device, dtype=torch.long) if labels is not None else None

        for i in range(batch_size):
            seq_len = merged_embeds_list[i].shape[0]
            padded_embeds[i, :seq_len] = merged_embeds_list[i]
            padded_attention[i, :seq_len] = merged_attention_list[i]
            if labels is not None:
                padded_labels[i, :seq_len] = merged_labels_list[i]

        # 4. Forward through LLM with autocast (auto float32 -> bfloat16)
        with _llm_autocast():  # [edge-eval patch 1/2] 原: torch.amp.autocast('cuda', dtype=torch.bfloat16)
            if sample_weights is not None and padded_labels is not None:
                # Per-sample weighted loss to handle class imbalance
                outputs = self.llm(inputs_embeds=padded_embeds, attention_mask=padded_attention)
                logits = outputs.logits  # (B, seq, vocab)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = padded_labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                loss_per_token = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(batch_size, -1)
                valid = (shift_labels != -100).float()
                per_sample_loss = (loss_per_token * valid).sum(-1) / valid.sum(-1).clamp(min=1)
                w = sample_weights.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
                outputs.loss = (per_sample_loss * w).mean()
            else:
                outputs = self.llm(
                    inputs_embeds=padded_embeds,
                    attention_mask=padded_attention,
                    labels=padded_labels,
                )

        # --- 视觉区域(逐环)检测头的辅助回归损失 ---
        # ring_scores 预测逐环【相对】强度, 监督目标来自 CCA 派生的 ring_target。
        outputs.lm_loss = outputs.loss
        outputs.aux_loss = None
        if self.use_region and ring_scores is not None and ring_target is not None \
                and outputs.loss is not None:
            rt = ring_target.to(ring_scores.device, dtype=ring_scores.dtype)
            aux = nn.functional.mse_loss(ring_scores, rt)
            outputs.aux_loss = aux
            outputs.loss = outputs.loss + self.aux_weight * aux

        # --- VFI 回归头的辅助损失 (可靠预测就来自这里) ---
        # vfi_pred ∈ [0,1] 对归一化目标 vfi_target(=VFI/100)。损失类型见 self.vfi_loss:
        #   mse       : 绝对值 MSE (旧行为, 右偏分布下塌成均值)
        #   huber     : 鲁棒回归 (delta=0.1 即 10 VFI 点), 缓右偏长尾对均值的拉扯
        #   rank      : 纯成对排序 (无法靠常数最小化, 只学「谁视野更好」的顺序)
        #   rankhuber : Huber 校准数值尺度 + rank_weight×排序正则 (推荐, 数值可读又保序)
        # ⚠️ 排序对是 batch 内所有清晰有序对; exclude 口径下全是有真实 VFI 的青光眼眼, 无需再滤 has_vf。
        if self.use_vfi and vfi_pred is not None and vfi_target is not None \
                and outputs.loss is not None:
            vt = vfi_target.to(vfi_pred.device, dtype=vfi_pred.dtype)
            if self.vfi_loss == 'mse':
                aux = nn.functional.mse_loss(vfi_pred, vt)
            elif self.vfi_loss == 'huber':
                aux = nn.functional.huber_loss(vfi_pred, vt, delta=0.1)
            elif self.vfi_loss == 'rank':
                aux = vfi_pairwise_rank_loss(vfi_pred, vt)
            elif self.vfi_loss == 'rankhuber':
                huber = nn.functional.huber_loss(vfi_pred, vt, delta=0.1)
                rank = vfi_pairwise_rank_loss(vfi_pred, vt)
                aux = huber + self.vfi_rank_weight * rank
            else:
                raise ValueError(f"未知 vfi_loss={self.vfi_loss}")
            outputs.aux_loss = aux
            outputs.loss = outputs.loss + self.vfi_aux_weight * aux
        return outputs

    @torch.no_grad()
    def predict_rings(self, eeg_signal):
        """仅返回逐环相对强度分数 (B, n_ring)；use_region=False 时返回 None。"""
        if not self.use_region:
            return None
        _, ring_scores, _ = self.encode_eeg(eeg_signal)
        return ring_scores

    @torch.no_grad()
    def predict_vfi(self, eeg_signal):
        """返回归一化 VFI 预测 (B,) ∈ [0,1]；use_vfi=False 时返回 None。
        评测时乘以 100 还原为 0~100 的 VFI。"""
        if not self.use_vfi:
            return None
        _, _, vfi_pred = self.encode_eeg(eeg_signal)
        return vfi_pred

    @torch.no_grad()
    def generate(self, eeg_signal, prompt_text, max_new_tokens=64, return_rings=False,
                 return_vfi=False):
        """Generate text response given EEG signal and text prompt.

        return_rings=True 时返回 (text, ring_scores)；return_vfi=True 时返回 (text, vfi_pred)；
        否则只返回 text。
        """
        device = eeg_signal.device

        # Encode EEG (+ 区域/VFI token) 和逐环分数/VFI 预测
        eeg_embeds, ring_scores, vfi_pred = self.encode_eeg(eeg_signal)

        # Tokenize prompt (should contain <eeg> placeholder)
        tokens = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = tokens.input_ids  # (1, seq_len)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Replace <eeg> with EEG embeddings
        eeg_pos = (input_ids[0] == self.eeg_token_id).nonzero(as_tuple=True)[0]
        if len(eeg_pos) > 0:
            pos = eeg_pos[0].item()
            before = text_embeds[0, :pos]
            after = text_embeds[0, pos + 1:]
            merged = torch.cat([before, eeg_embeds[0], after], dim=0).unsqueeze(0)
            attn_mask = torch.ones(1, merged.shape[1], device=device, dtype=torch.long)
        else:
            merged = text_embeds
            attn_mask = tokens.attention_mask

        with _llm_autocast():  # [edge-eval patch 1/2] 原: torch.amp.autocast('cuda', dtype=torch.bfloat16)
            outputs = self.llm.generate(
                inputs_embeds=merged,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if return_rings:
            return text, ring_scores
        if return_vfi:
            return text, vfi_pred
        return text
