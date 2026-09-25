"""
Preprocess Glaucoma OpenBCI SSVEP EEG data into LMDB format.

Labels are read from the xlsx file:
  EEG-OpenBCI-Participants-Last.xlsx (columns: Days, Participant, eyes, ECG-filename, Glaucoma)

Rules:
  - Skip rows with NaN Days or NaN ECG-filename
  - Glaucoma="Yes" -> label 1, Glaucoma="No" -> label 0
  - Each session = 1 participant's 1 eye

Preprocessing pipeline:
  1. Load raw OpenBCI txt (250Hz, columns 2-7 = 6 valid EEG channels)
  2. Trim first/last 10 seconds
  3. Resample 250Hz -> 200Hz
  4. Bandpass filter 0.3-75Hz
  5. 50Hz notch filter
  6. Segment into 5-second windows -> (6, 5, 200)
  7. Save to LMDB

Usage:
  EEGLLAVA_RAW_DIR=<raw dir> EEGLLAVA_LABEL_SHEET=<participants.csv|xlsx> \
  EEGLLAVA_LMDB=<output lmdb dir> python preprocessing/preprocess_glaucoma_openbci.py
"""

import os
import glob
import numpy as np
import pandas as pd
import mne
import lmdb
import pickle

# --- Configuration ---
# [release] server paths replaced by environment variables
data_root = os.environ.get('EEGLLAVA_RAW_DIR', 'data/raw')
xlsx_path = os.environ.get('EEGLLAVA_LABEL_SHEET', 'data/metadata/participants.csv')
output_dir = os.environ.get('EEGLLAVA_LMDB', 'data/processed_lmdb')

srate_original = 250
srate_target = 200
patch_size = 200
num_patches = 5
segment_points = patch_size * num_patches  # 1000

valid_data_columns = list(range(1, 7))  # 6 channels
ch_names = ['EXG0', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5']

l_freq = 0.3
h_freq = 75.0
notch_freq = 50.0

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# --- End of Configuration ---


def load_openbci_txt(filepath):
    df = pd.read_csv(
        filepath, comment='%', header=None, delimiter=',',
        skipinitialspace=True, low_memory=False
    )
    df_data = df.iloc[1:]
    eeg_data = df_data.iloc[:, valid_data_columns].astype(float).to_numpy()
    return eeg_data


def preprocess_eeg(eeg_data):
    # Trim first and last 10 seconds
    trim_samples = 10 * srate_original
    if eeg_data.shape[0] > 2 * trim_samples:
        eeg_data = eeg_data[trim_samples:-trim_samples, :]

    data_volts = eeg_data.T * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=srate_original, ch_types='eeg')
    raw = mne.io.RawArray(data_volts, info, verbose=False)
    raw.resample(srate_target, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(notch_freq, verbose=False)
    processed = raw.get_data().T * 1e6
    return processed


def segment_eeg(eeg_data):
    n_samples, n_channels = eeg_data.shape
    remainder = n_samples % segment_points
    if remainder != 0:
        eeg_data = eeg_data[:-remainder, :]
    n_segments = len(eeg_data) // segment_points
    if n_segments == 0:
        return np.array([])
    segments = eeg_data.reshape(n_segments, num_patches, patch_size, n_channels)
    segments = segments.transpose(0, 3, 1, 2)
    return segments


def load_xlsx_labels(xlsx_path):
    """
    Parse xlsx to get list of valid sessions with labels.
    Returns: list of dicts: {day, participant, eye, session_dir, label}
    """
    df = pd.read_csv(xlsx_path) if str(xlsx_path).lower().endswith('.csv') else pd.read_excel(xlsx_path)  # [release]
    records = []

    for _, row in df.iterrows():
        # Skip NaN rows (separators)
        if pd.isna(row['Days']) or pd.isna(row['ECG-filename']):
            continue

        day = str(row['Days']).strip()

        session_dir = str(row['ECG-filename']).strip()
        glaucoma = str(row['Glaucoma']).strip()
        label = 1 if glaucoma.lower() == 'yes' else 0  # [release] was == 'Yes'
        participant = int(row['Participant'])
        eye = str(row['eyes']).strip()

        records.append({
            'day': day,
            'participant': participant,
            'eye': eye,
            'session_dir': session_dir,
            'label': label,
        })

    return records


def find_txt_in_session(data_root, day, session_dir):
    """Find the txt file strictly in the day/session directory specified by xlsx."""
    session_path = os.path.join(data_root, day, session_dir)
    if os.path.isdir(session_path):
        txt_files = glob.glob(os.path.join(session_path, '*.txt'))
        if txt_files:
            return txt_files[0]
    return None


def main():
    # Load labels from xlsx
    records = load_xlsx_labels(xlsx_path)
    print(f"Loaded {len(records)} valid sessions from xlsx")

    n_glaucoma = sum(1 for r in records if r['label'] == 1)
    n_healthy = sum(1 for r in records if r['label'] == 0)
    print(f"  Glaucoma (Yes): {n_glaucoma} sessions")
    print(f"  Healthy  (No):  {n_healthy} sessions")

    # Process each session
    all_samples = {}
    session_info = []  # (session_id, day, participant, label, n_segments)

    for rec in records:
        txt_file = find_txt_in_session(data_root, rec['day'], rec['session_dir'])
        if txt_file is None:
            print(f"  SKIP: {rec['day']}/{rec['session_dir']} - txt not found")
            continue

        session_id = f"{rec['day']}_P{rec['participant']}_{rec['eye']}_{rec['session_dir']}"
        label_str = "glaucoma" if rec['label'] == 1 else "healthy"

        try:
            eeg_data = load_openbci_txt(txt_file)
            if eeg_data.shape[0] < segment_points:
                print(f"  SKIP {session_id}: too short ({eeg_data.shape[0]} samples)")
                continue

            eeg_data = preprocess_eeg(eeg_data)
            segments = segment_eeg(eeg_data)
            if len(segments) == 0:
                continue

            n_seg = len(segments)
            for i, sample in enumerate(segments):
                key = f"{session_id}_{i}"
                all_samples[key] = {
                    'sample': sample.astype(np.float32),
                    'label': rec['label'],
                }

            session_info.append((session_id, rec['day'], rec['participant'],
                                 rec['label'], n_seg))
            print(f"  {session_id} [{label_str}]: {n_seg} segments")

        except Exception as e:
            print(f"  ERROR {session_id}: {e}")
            continue

    # --- Summary ---
    n_healthy_sess = sum(1 for _, _, _, l, _ in session_info if l == 0)
    n_glaucoma_sess = sum(1 for _, _, _, l, _ in session_info if l == 1)
    n_seg_healthy = sum(n for _, _, _, l, n in session_info if l == 0)
    n_seg_glaucoma = sum(n for _, _, _, l, n in session_info if l == 1)
    print(f"\n{'='*60}")
    print(f"Sessions:  healthy={n_healthy_sess}, glaucoma={n_glaucoma_sess}, "
          f"total={len(session_info)}")
    print(f"Segments:  healthy={n_seg_healthy}, glaucoma={n_seg_glaucoma}, "
          f"total={len(all_samples)}")

    if not all_samples:
        print("ERROR: No data!")
        return

    # --- Train/Val/Test split (by eye, each session = one eye = one unit) ---
    np.random.seed(42)

    # Each session is one eye, treat as independent unit
    healthy_eyes = [info for info in session_info if info[3] == 0]
    glaucoma_eyes = [info for info in session_info if info[3] == 1]
    np.random.shuffle(healthy_eyes)
    np.random.shuffle(glaucoma_eyes)

    def split_eyes(eyes):
        n = len(eyes)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        return (eyes[:n_train],
                eyes[n_train:n_train + n_val],
                eyes[n_train + n_val:])

    h_train, h_val, h_test = split_eyes(healthy_eyes)
    g_train, g_val, g_test = split_eyes(glaucoma_eyes)

    if not h_test:
        h_test = h_val
    if not g_test:
        g_test = g_val

    def eyes_to_keys(eyes):
        keys = []
        for sid, _, _, _, n_seg in eyes:
            keys.extend([f"{sid}_{i}" for i in range(n_seg)])
        return keys

    dataset_keys = {
        'train': eyes_to_keys(h_train + g_train),
        'val': eyes_to_keys(h_val + g_val),
        'test': eyes_to_keys(h_test + g_test),
    }

    print(f"\nSplit (by eye, stratified):")
    print(f"  Train: {len(h_train)} healthy + {len(g_train)} glaucoma eyes "
          f"-> {len(dataset_keys['train'])} segments")
    print(f"  Val:   {len(h_val)} healthy + {len(g_val)} glaucoma eyes "
          f"-> {len(dataset_keys['val'])} segments")
    print(f"  Test:  {len(h_test)} healthy + {len(g_test)} glaucoma eyes "
          f"-> {len(dataset_keys['test'])} segments")

    # --- Save to LMDB ---
    # Remove old LMDB if exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"\nRemoved old LMDB: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to LMDB: {output_dir}")
    db = lmdb.open(output_dir, map_size=4294967296)

    txn = db.begin(write=True)
    for key, data_dict in all_samples.items():
        txn.put(key=key.encode(), value=pickle.dumps(data_dict))
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset_keys))
    txn.commit()
    db.close()

    # Verification
    sample_key = list(all_samples.keys())[0]
    sample = all_samples[sample_key]
    print(f"\nVerification:")
    print(f"  Key: {sample_key}")
    print(f"  Shape: {sample['sample'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Value range: [{sample['sample'].min():.2f}, {sample['sample'].max():.2f}]")
    print(f"\nDone! Output: {output_dir}")


if __name__ == '__main__':
    main()
