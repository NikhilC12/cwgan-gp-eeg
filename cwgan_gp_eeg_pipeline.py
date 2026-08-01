# =============================================================================
# Epileptic Seizure Detection — Complete Research Pipeline
# Conditional WGAN-GP with Temporal Smoothness Regularisation
#
# Datasets:
#   1. Bonn University Dataset (Andrzejak et al., 2001) — primary, 80/20 split
#   2. CHB-MIT Scalp EEG (PhysioNet)                    — external validation
#
# FEATURE SPACE DESIGN (single source of truth — must match paper Section III.A.2):
#   Both classical ML and deep learning models are trained and evaluated in ONE
#   consistent representation: the MinMaxScaler fit exclusively on Bonn TRAINING
#   data, in range [-1, 1]. CHB-MIT windows are extracted in raw microvolt
#   amplitude, then transformed with this SAME fitted scaler (transform only,
#   never re-fit) before being used anywhere downstream — classical ML, deep
#   learning, GAN training/generation, and all evaluation metrics. There is no
#   separate "raw amplitude" evaluation path for classical ML. If you change
#   this design, update Section III.A.2 of the paper in the same commit.
#
# REPRODUCIBILITY NOTES:
#   - Bonn dataset: public, UCI ML Repository / Andrzejak et al. (2001).
#   - CHB-MIT dataset: public, PhysioNet (Shoeb & Guttag, 2010), requires
#     PhysioNet credentialed access terms (no special credentialing needed,
#     but cite the data use agreement in the paper's data availability
#     statement).
#   - Set DATA_DIR and CHBMIT_DIR below (or via environment variables) to
#     point at your local copies. Do not commit raw data or personal cloud
#     storage paths to the repository.
# =============================================================================

# ── 0. IMPORTS & GLOBAL SEEDS ────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch, resample
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import svm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Concatenate, Reshape,
                                     UpSampling1D, Conv1D, BatchNormalization,
                                     LeakyReLU, Flatten, Multiply, Add, Lambda)
import keras
from tensorflow.keras.models import Model
import joblib
import mne

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── 0b. DATA PATHS (edit these or set as environment variables) ─────────────
# NOTE FOR REVIEWERS: no personal cloud storage paths are used. Set these two
# variables to point at your local copies of the public Bonn CSV and the
# CHB-MIT .edf files before running.
DATA_DIR   = os.environ.get("EEG_DATA_DIR", "./data")            # contains data.csv (Bonn)
CHBMIT_DIR = os.environ.get("EEG_CHBMIT_DIR", "./data/chbmit")    # contains chbNN_XX.edf files
DATA_PATH  = os.path.join(DATA_DIR, "data.csv")

def chbmit_path(filename):
    return os.path.join(CHBMIT_DIR, filename)

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
NOISE_DIM             = 256
NUM_CLASSES           = 2
NUM_FEATURES          = 178
BATCH_SIZE            = 64
TRAIN_STEPS           = 10000
LAMBDA_GP             = 10
LAMBDA_SMOOTH         = 0.01
LAMBDA_SPEC           = 0.05
EMA_DECAY             = 0.999
LR_START              = 1e-4
LR_END                = 1e-5
LR_DECAY_FROM         = 0.70
CRITIC_STEPS_EARLY    = 5
CRITIC_STEPS_LATE     = 3
CRITIC_ADAPT_THRESH   = 0.01
CRITIC_ADAPT_WINDOW   = 200
EARLY_STOP_PATIENCE   = 2500
EARLY_STOP_EVAL_EVERY = 250

CHB_NATIVE_FS = 256   # CHB-MIT native sampling rate (Hz)

# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================
print("=" * 60)
print("1. Loading and preprocessing data")
print("=" * 60)

# ── 2a. Primary dataset (Bonn University) ────────────────────────────────────
dataset   = pd.read_csv(DATA_PATH)
epilepsy  = pd.DataFrame(dataset)

print(f"[Bonn] Dataset shape: {epilepsy.shape}")
print(f"[Bonn] Class distribution:\n{epilepsy['y'].value_counts()}")

epilepsy['y'] = epilepsy['y'].replace([2, 3, 4, 5], 0)
if 'Unnamed: 0' in epilepsy.columns:
    epilepsy = epilepsy.drop(columns='Unnamed: 0')

X = epilepsy.drop(columns='y').values
y = epilepsy['y'].values
print(f"\n[Bonn] Binary: seizure={y.sum()}, non-seizure={(y==0).sum()}")

# ── 2b. Train/test split (Bonn) — MUST happen before scaler is fit ───────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"\n[Bonn] Train: {X_train.shape}, Test (80/20): {X_test.shape}")

# ── 2c. Fit scaler on Bonn TRAINING data only, then save it ──────────────────
#   This is the ONE reference frame used everywhere downstream (see docstring).
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
joblib.dump(scaler, 'bonn_minmax_scaler.pkl')
print("\n[Scaler] Fit on Bonn training data and saved to bonn_minmax_scaler.pkl")
print(f"[Scaler] Data range after transform: "
      f"[{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")

# =============================================================================
# 3. CHB-MIT EXTERNAL VALIDATION — RAW EXTRACTION + BONN SCALER (transform only)
# =============================================================================
print("\n" + "=" * 60)
print("2. Extracting CHB-MIT external validation set")
print("=" * 60)

def extract_raw_windows(file_path, start_time, num_windows,
                        native_fs=CHB_NATIVE_FS, target_samples=NUM_FEATURES):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    found_ch = []
    for label in ('FP1-F7', 'T7-P7'):
        found_ch = [ch for ch in raw.ch_names if label in ch]
        if found_ch:
            break
    if not found_ch:
        raise ValueError(f"No usable channel in {file_path}")

    raw.pick_channels([found_ch[0]])
    data = raw.get_data()[0]

    # ── UNIT FIX: MNE returns volts, Bonn dataset is in microvolts ───────────
    data = data * 1e6   # V → μV

    windows = []
    for i in range(num_windows):
        s = int((start_time + i) * native_fs)
        e = s + native_fs
        if e > len(data):
            print(f"  [INFO] End of file at window {i} in {file_path}")
            break
        windows.append(resample(data[s:e], target_samples))
    return np.array(windows)


# ── Seizure windows (timings from CHB-MIT .seizures annotation files) ─────────
print("  Extracting seizure windows …")
seizure_data_1 = extract_raw_windows(chbmit_path("chb01_03.edf"), start_time=2996,  num_windows=40)
seizure_data_2 = extract_raw_windows(chbmit_path("chb03_01.edf"), start_time=362,   num_windows=52)
seizure_data_3 = extract_raw_windows(chbmit_path("chb05_06.edf"), start_time=417,   num_windows=115)
seizure_data_4 = extract_raw_windows(chbmit_path("chb07_12.edf"), start_time=4920,  num_windows=86)
seizure_data_5 = extract_raw_windows(chbmit_path("chb08_02.edf"), start_time=2670,  num_windows=167)

# ── Non-seizure windows (confirmed interictal, ≥30 s from any seizure) ────────
print("  Extracting non-seizure windows …")
nonseizure_data_1 = extract_raw_windows(chbmit_path("chb01_03.edf"), start_time=1000, num_windows=368)
nonseizure_data_2 = extract_raw_windows(chbmit_path("chb03_01.edf"), start_time=1000, num_windows=368)
nonseizure_data_3 = extract_raw_windows(chbmit_path("chb05_06.edf"), start_time=1000, num_windows=368)
nonseizure_data_4 = extract_raw_windows(chbmit_path("chb07_12.edf"), start_time=1000, num_windows=368)
nonseizure_data_5 = extract_raw_windows(chbmit_path("chb08_02.edf"), start_time=1000, num_windows=368)

# ── Concatenate raw windows ───────────────────────────────────────────────────
X_chbmit_seizures    = np.concatenate([seizure_data_1, seizure_data_2, seizure_data_3,
                                        seizure_data_4, seizure_data_5], axis=0)
X_chbmit_nonseizures = np.concatenate([nonseizure_data_1, nonseizure_data_2, nonseizure_data_3,
                                        nonseizure_data_4, nonseizure_data_5], axis=0)

X_chbmit_raw = np.concatenate([X_chbmit_seizures, X_chbmit_nonseizures], axis=0)
y_chbmit     = np.concatenate([
    np.ones(len(X_chbmit_seizures),    dtype=int),   # 1 = ictal
    np.zeros(len(X_chbmit_nonseizures), dtype=int),  # 0 = interictal
], axis=0)

# ── Remove flat windows (zero std → uninformative) ────────────────────────────
is_flat      = np.std(X_chbmit_raw, axis=1) == 0
X_chbmit_raw = X_chbmit_raw[~is_flat]
y_chbmit     = y_chbmit[~is_flat]
print(f"  Flat windows removed: {is_flat.sum()}")

# ── Apply Bonn scaler (transform only — never refit) ──────────────────────────
#   This is the ONLY feature space used for CHB-MIT anywhere in this pipeline,
#   for both classical ML and deep learning. See module docstring.
X_chbmit = scaler.transform(X_chbmit_raw).astype(np.float32)

print(f"\n[CHB-MIT] Samples after cleaning : {len(y_chbmit)}")
print(f"[CHB-MIT] Seizure windows        : {y_chbmit.sum()}")
print(f"[CHB-MIT] Non-seizure windows    : {(y_chbmit==0).sum()}")
print(f"[CHB-MIT] Value range after Bonn scaler: "
      f"[{X_chbmit.min():.4f}, {X_chbmit.max():.4f}]")

assert X_chbmit.shape[1] == NUM_FEATURES, \
    f"Feature mismatch: CHB-MIT has {X_chbmit.shape[1]}, expected {NUM_FEATURES}"

print(f"\n[Bonn]   Train: {X_train.shape}, Test: {X_test.shape}")
print(f"[CHB-MIT] External validation (Bonn-scaled): {X_chbmit.shape}")

# ── 2d. Visualise example EEG waveforms (Bonn, 5 original classes) ───────────
raw_df       = pd.read_csv(DATA_PATH)
feature_cols = [c for c in raw_df.columns if c.startswith('X')]

fig, axes = plt.subplots(1, 5, figsize=(18, 3), sharey=False)
class_names = {1: 'Ictal (seizure)', 2: 'Interictal (tumour)',
               3: 'Interictal (healthy)', 4: 'Eyes open', 5: 'Eyes closed'}
colors = ['red', 'blue', 'green', 'orange', 'black']
for idx, (cls, name) in enumerate(class_names.items()):
    row = raw_df[raw_df['y'] == cls].iloc[0][feature_cols].values.astype(float)
    axes[idx].plot(row, color=colors[idx], linewidth=0.8)
    axes[idx].set_title(name, fontsize=8)
    axes[idx].set_xlabel('Time step')
    if idx == 0:
        axes[idx].set_ylabel('Amplitude (μV)')
plt.suptitle('Representative EEG Segments by Brain State (Bonn)', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_eeg_waveforms.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 2e. Visualise CHB-MIT examples (in raw amplitude, before scaling) ─────────
fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=False)
chbmit_plot_specs = [(1,'Ictal','red'), (1,'Ictal','darkred'),
                     (0,'Non-ictal','steelblue'), (0,'Non-ictal','navy')]
counts = {0: 0, 1: 0}
for plot_idx, (lbl, title, col) in enumerate(chbmit_plot_specs):
    mask = np.where(y_chbmit == lbl)[0]
    row  = X_chbmit_raw[mask[counts[lbl]]]
    counts[lbl] += 1
    axes[plot_idx].plot(row, color=col, linewidth=0.8)
    axes[plot_idx].set_title(f'CHB-MIT {title} {counts[lbl]}', fontsize=9)
    axes[plot_idx].set_xlabel('Time step')
    if plot_idx == 0:
        axes[plot_idx].set_ylabel('Amplitude (μV)')
plt.suptitle('Representative CHB-MIT EEG Segments (raw amplitude)', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_chbmit_waveforms.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. GAN DATA PREP
# =============================================================================
print("\n" + "=" * 60)
print("3. Training Improved CWGAN-GP")
print("=" * 60)

X_test_scaled = scaler.transform(X_test).astype(np.float32)

ohe            = OneHotEncoder(sparse_output=False)
y_train_ohe    = ohe.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
real_ictal_scaled = X_train_scaled[y_train == 1]

# =============================================================================
# 5–13. GAN ARCHITECTURE
# =============================================================================

def build_generator():
    noise_in = Input(shape=(NOISE_DIM,), name='noise')
    label_in = Input(shape=(NUM_CLASSES,), name='label')
    x = Concatenate()([noise_in, label_in])
    x = Dense(256 * 89)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((89, 256))(x)
    x = UpSampling1D()(x)
    x = Conv1D(128, 5, padding='same')(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    x = Conv1D(64,  5, padding='same')(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    x = Conv1D(32,  5, padding='same')(x); x = LeakyReLU(0.2)(x)
    x = Conv1D(1,   3, padding='same', activation='tanh')(x)
    output = Flatten()(x)
    return Model([noise_in, label_in], output, name='Generator')

def build_critic():
    data_in  = Input(shape=(NUM_FEATURES,), name='eeg')
    label_in = Input(shape=(NUM_CLASSES,),  name='label')
    x = Reshape((NUM_FEATURES, 1))(data_in)
    x = Conv1D(64,  5, strides=2, padding='same')(x); x = LeakyReLU(0.2)(x)
    x = Conv1D(128, 5, strides=2, padding='same')(x); x = LeakyReLU(0.2)(x)
    x = Conv1D(256, 5, strides=2, padding='same')(x); x = LeakyReLU(0.2)(x)
    x = Conv1D(512, 5, strides=2, padding='same')(x); x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    features = Dense(128)(x); features = LeakyReLU(0.2)(features)
    label_embedding  = Dense(128, use_bias=False)(label_in)
    projected        = Multiply()([features, label_embedding])
    projection_score = Lambda(lambda t: keras.ops.sum(t, axis=1, keepdims=True))(projected)
    unconditional_score = Dense(1)(features)
    output = Add()([unconditional_score, projection_score])
    return Model([data_in, label_in], output, name='Critic')

generator     = build_generator()
critic        = build_critic()
ema_generator = build_generator()
ema_generator.set_weights(generator.get_weights())

def update_ema(live_model, ema_model, decay=EMA_DECAY):
    for live_w, ema_w in zip(live_model.weights, ema_model.weights):
        ema_w.assign(decay * ema_w + (1.0 - decay) * live_w)

class LinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_start, lr_end, total_steps, decay_from):
        super().__init__()
        self.lr_start    = float(lr_start)
        self.lr_end      = float(lr_end)
        self.total_steps = float(total_steps)
        self.decay_start = float(int(decay_from * total_steps))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        frac = tf.maximum(0.0, step - self.decay_start) / tf.maximum(
                   1.0, self.total_steps - self.decay_start)
        frac = tf.minimum(frac, 1.0)
        return self.lr_start + frac * (self.lr_end - self.lr_start)

    def get_config(self):
        return {'lr_start': self.lr_start, 'lr_end': self.lr_end,
                'total_steps': self.total_steps, 'decay_start': self.decay_start}

lr_schedule = LinearDecaySchedule(LR_START, LR_END, TRAIN_STEPS, LR_DECAY_FROM)
g_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.0, beta_2=0.9)
c_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.0, beta_2=0.9)

@tf.function
def spectral_loss(real_batch, fake_batch):
    real_fft  = tf.abs(tf.signal.rfft(real_batch))
    fake_fft  = tf.abs(tf.signal.rfft(fake_batch))
    real_norm = real_fft / (tf.reduce_sum(real_fft, axis=1, keepdims=True) + 1e-8)
    fake_norm = fake_fft / (tf.reduce_sum(fake_fft, axis=1, keepdims=True) + 1e-8)
    return tf.reduce_mean(tf.square(real_norm - fake_norm))

def gradient_penalty(real_batch, fake_batch, labels_batch):
    alpha        = tf.random.uniform([BATCH_SIZE, 1], 0.0, 1.0)
    interpolated = real_batch + alpha * (fake_batch - real_batch)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated, labels_batch], training=True)
    grads = gp_tape.gradient(pred, interpolated)
    norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
    return tf.reduce_mean((norm - 1.0) ** 2)

@tf.function
def train_step(real_data, labels, spec_batch, critic_steps):
    for _ in range(critic_steps):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        with tf.GradientTape() as c_tape:
            fake_data  = generator([noise, labels], training=True)
            real_score = critic([real_data, labels], training=True)
            fake_score = critic([fake_data, labels], training=True)
            gp         = gradient_penalty(real_data, fake_data, labels)
            c_loss     = (tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
                          + LAMBDA_GP * gp)
        c_grads = c_tape.gradient(c_loss, critic.trainable_variables)
        c_opt.apply_gradients(zip(c_grads, critic.trainable_variables))

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as g_tape:
        fake_data      = generator([noise, labels], training=True)
        fake_score     = critic([fake_data, labels], training=True)
        g_loss_adv     = -tf.reduce_mean(fake_score)
        temporal_diffs = fake_data[:, 1:] - fake_data[:, :-1]
        smoothness_reg = tf.reduce_mean(tf.square(temporal_diffs))
        spec_loss      = spectral_loss(spec_batch, fake_data)
        g_loss = g_loss_adv + LAMBDA_SMOOTH * smoothness_reg + LAMBDA_SPEC * spec_loss
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    w_estimate = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
    return c_loss, g_loss, w_estimate

def compute_mmd_quick(n_synth=500):
    noise  = np.random.normal(0, 1, (n_synth, NOISE_DIM)).astype(np.float32)
    labels = tf.one_hot(np.ones(n_synth, dtype=int), depth=NUM_CLASSES)
    synth  = ema_generator.predict([noise, labels], verbose=0)
    ref    = real_ictal_scaled[:n_synth]
    XX = rbf_kernel(ref, ref); YY = rbf_kernel(synth, synth); XY = rbf_kernel(ref, synth)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nTraining improved CWGAN-GP for up to {TRAIN_STEPS} steps...")

critic_losses    = []
generator_losses = []
wasserstein_hist = []
best_mmd         = float('inf')
best_gen_weights = None
no_improve_steps = 0

for step in range(TRAIN_STEPS):
    idx          = np.random.randint(0, X_train_scaled.shape[0], BATCH_SIZE)
    real_batch   = X_train_scaled[idx]
    labels_batch = y_train_ohe[idx]

    if len(wasserstein_hist) < CRITIC_ADAPT_WINDOW:
        n_critic = CRITIC_STEPS_EARLY
    else:
        delta    = abs(np.mean(wasserstein_hist[-CRITIC_ADAPT_WINDOW:]) -
                       np.mean(wasserstein_hist[-CRITIC_ADAPT_WINDOW//2:]))
        n_critic = CRITIC_STEPS_LATE if delta < CRITIC_ADAPT_THRESH else CRITIC_STEPS_EARLY

    ictal_idx  = np.random.choice(np.where(y_train == 1)[0], BATCH_SIZE)
    spec_batch = X_train_scaled[ictal_idx]

    c_loss, g_loss, w_est = train_step(real_batch, labels_batch, spec_batch, n_critic)
    critic_losses.append(float(c_loss))
    generator_losses.append(float(g_loss))
    wasserstein_hist.append(float(w_est))
    update_ema(generator, ema_generator)

    if (step + 1) % EARLY_STOP_EVAL_EVERY == 0:
        mmd_val = compute_mmd_quick()
        if mmd_val < best_mmd:
            best_mmd         = mmd_val
            best_gen_weights = ema_generator.get_weights()
            no_improve_steps = 0
        else:
            no_improve_steps += EARLY_STOP_EVAL_EVERY
        print(f"  Step {step+1:5d} | C: {c_loss:.4f} | G: {g_loss:.4f} | "
              f"MMD: {mmd_val:.6f} | best: {best_mmd:.6f} | n_crit: {n_critic}")
        if no_improve_steps >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop at step {step+1}.")
            break
    elif step % 500 == 0:
        print(f"  Step {step:5d} | C: {c_loss:.4f} | G: {g_loss:.4f} | n_crit: {n_critic}")

if best_gen_weights is not None:
    ema_generator.set_weights(best_gen_weights)
    print(f"\nRestored best EMA checkpoint (MMD={best_mmd:.6f})")

# ── Save trained models so future work (TSTR, more samples, extra McNemar
#    comparisons, different seeds) doesn't require a full GAN retrain ───────
ema_generator.save('ema_generator.keras')
generator.save('generator_final.keras')
critic.save('critic_final.keras')
print("Saved ema_generator.keras, generator_final.keras, critic_final.keras")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_samples(n_samples, target_class=1):
    """Generate from EMA generator. Returns (X_scaled, y) in Bonn [-1,1] space."""
    noise  = np.random.normal(0, 1, (n_samples, NOISE_DIM)).astype(np.float32)
    labels = tf.one_hot(np.full(n_samples, target_class, dtype=int), depth=NUM_CLASSES)
    synth_scaled = ema_generator.predict([noise, labels], verbose=0)
    return synth_scaled, np.full(n_samples, target_class, dtype=int)

def evaluate_classifier(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    TP = int(((y_pred==1)&(y_te==1)).sum()); FP = int(((y_pred==1)&(y_te==0)).sum())
    FN = int(((y_pred==0)&(y_te==1)).sum()); TN = int(((y_pred==0)&(y_te==0)).sum())
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = TP/(TP+0.5*FP+0.5*FN) if (TP+0.5*FP+0.5*FN)>0 else 0.0
    acc  = accuracy_score(y_te, y_pred)
    return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1': f1, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'y_pred': y_pred, 'model': model}

def score_on_set(model, X_te, y_te):
    y_pred = model.predict(X_te)
    TP = int(((y_pred==1)&(y_te==1)).sum()); FP = int(((y_pred==1)&(y_te==0)).sum())
    FN = int(((y_pred==0)&(y_te==1)).sum()); TN = int(((y_pred==0)&(y_te==0)).sum())
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = TP/(TP+0.5*FP+0.5*FN) if (TP+0.5*FP+0.5*FN)>0 else 0.0
    return {'Accuracy': accuracy_score(y_te, y_pred), 'Precision': prec,
            'Recall': rec, 'F1': f1, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'y_pred': y_pred}

def make_table(results_list, label):
    rows = [{k: v for k, v in r.items() if k not in ('y_pred','model')}
            for r in results_list]
    df = pd.DataFrame(rows)
    print(f"\n── {label} ──")
    print(df[['Model','Accuracy','Precision','Recall','F1','TP','FP','FN','TN']].to_string(index=False))
    return df

# =============================================================================
# CLASSIFIERS — all trained/evaluated in the SAME Bonn-scaler [-1,1] space
# for both Bonn test and CHB-MIT (single feature-space design, see docstring).
# =============================================================================

CLASSIFIERS = {
    'Logistic Regression':    LogisticRegression(max_iter=1000, random_state=SEED),
    'Ridge Regression':       RidgeClassifier(max_iter=1000),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=SEED),
    'Decision Tree':          DecisionTreeClassifier(random_state=SEED),
    'Support Vector Machine': svm.SVC(max_iter=10000, probability=True, random_state=SEED),
    'Linear SVM':             svm.LinearSVC(max_iter=10000, random_state=SEED),
}

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================
print("\n" + "=" * 60)
print("4. Generating synthetic EEG samples")
print("=" * 60)

n_ictal_train = int((y_train == 1).sum())
n_non_ictal   = int((y_train == 0).sum())
n_to_generate = n_non_ictal - n_ictal_train

X_synth_scaled, y_synth = generate_samples(n_to_generate, target_class=1)
print(f"Generated {n_to_generate} synthetic ictal samples (scaled space)")

# =============================================================================
# DISTRIBUTION METRICS
# =============================================================================
print("\n" + "=" * 60)
print("5. Distribution similarity metrics")
print("=" * 60)

real_ictal_all_scaled = X_train_scaled[y_train == 1]
real_ictal_all_raw    = X_train[y_train == 1]
synth_eval_scaled, _  = generate_samples(len(real_ictal_all_scaled), target_class=1)
synth_eval_raw        = scaler.inverse_transform(synth_eval_scaled)

def compute_js_divergence(real_raw, synth_raw, fs=178):
    _, p_real  = welch(real_raw,  fs=fs, axis=1)
    _, p_synth = welch(synth_raw, fs=fs, axis=1)
    d_real  = np.mean(p_real,  axis=0); d_real  /= d_real.sum()
    d_synth = np.mean(p_synth, axis=0); d_synth /= d_synth.sum()
    return float(jensenshannon(d_real, d_synth))

def compute_mmd(real_s, synth_s):
    XX = rbf_kernel(real_s, real_s); YY = rbf_kernel(synth_s, synth_s)
    XY = rbf_kernel(real_s, synth_s)
    return float(XX.mean() + YY.mean() - 2*XY.mean())

def compute_vendi_score(synth_array):
    X_norm  = synth_array / (np.linalg.norm(synth_array, axis=1, keepdims=True) + 1e-8)
    K       = X_norm @ X_norm.T
    K       = (K + K.T) / 2
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 0]
    eigvals /= eigvals.sum()
    return float(np.exp(-np.sum(eigvals * np.log(eigvals + 1e-12))))

js_div = compute_js_divergence(real_ictal_all_raw, synth_eval_raw)
mmd    = compute_mmd(real_ictal_all_scaled, synth_eval_scaled)
synth_vendi, _ = generate_samples(500, target_class=1)
vs     = compute_vendi_score(synth_vendi)

print(f"  JS Divergence : {js_div:.4f}  (target <0.07)")
print(f"  MMD           : {mmd:.8f}")
print(f"  Vendi Score   : {vs:.2f}")

# =============================================================================
# AUGMENTED DATASET (scaled space throughout)
# =============================================================================
X_aug_scaled = np.vstack([X_train_scaled, X_synth_scaled])
y_aug        = np.concatenate([y_train, y_synth])
print(f"\nAugmented set: {X_aug_scaled.shape[0]} samples  "
      f"(ictal: {(y_aug==1).sum()}, non-ictal: {(y_aug==0).sum()})")

# =============================================================================
# GAUSSIAN NOISE BASELINE
# =============================================================================
def add_gaussian_noise(X_tr, y_tr, ratio=0.75, noise_std=0.05, seed=SEED):
    rng     = np.random.default_rng(seed)
    n       = int(len(X_tr) * ratio)
    idx     = rng.integers(0, len(X_tr), n)
    X_noisy = X_tr[idx] + rng.normal(0, noise_std, (n, X_tr.shape[1]))
    return np.vstack([X_tr, X_noisy]), np.concatenate([y_tr, y_tr[idx]])

X_train_gauss, y_train_gauss = add_gaussian_noise(X_train_scaled, y_train)

gauss_results = []
for name in ['Random Forest', 'Decision Tree']:
    clf = (RandomForestClassifier(n_estimators=100, random_state=SEED)
           if name == 'Random Forest' else DecisionTreeClassifier(random_state=SEED))
    r = evaluate_classifier(f"{name} (Gaussian)", clf,
                            X_train_gauss, X_test_scaled, y_train_gauss, y_test)
    gauss_results.append(r)

gauss_chbmit_results = []
for r in gauss_results:
    s = score_on_set(r['model'], X_chbmit, y_chbmit)
    gauss_chbmit_results.append({'Model': r['Model'], **s})

# =============================================================================
# BASELINE CLASSIFIERS
# =============================================================================
baseline_results = []
for name, clf in CLASSIFIERS.items():
    r = evaluate_classifier(name, clf, X_train_scaled, X_test_scaled, y_train, y_test)
    baseline_results.append(r)

baseline_chbmit_results = []
for r in baseline_results:
    s = score_on_set(r['model'], X_chbmit, y_chbmit)
    baseline_chbmit_results.append({'Model': r['Model'], **s})

# =============================================================================
# CWGAN-GP AUGMENTED CLASSIFIERS
# =============================================================================
cgan_results = []
aug_models   = {}
for name, clf_template in CLASSIFIERS.items():
    r = evaluate_classifier(f"{name} (CWGAN-GP)", clf_template,
                            X_aug_scaled, X_test_scaled, y_aug, y_test)
    cgan_results.append(r)
    aug_models[name] = r['model']

cgan_chbmit_results = []
for r in cgan_results:
    s = score_on_set(r['model'], X_chbmit, y_chbmit)
    cgan_chbmit_results.append({'Model': r['Model'], **s})

make_table(baseline_results + gauss_results + cgan_results,
           "Full Comparison — Bonn 80/20 Test Set")
make_table(baseline_chbmit_results + gauss_chbmit_results + cgan_chbmit_results,
           "Full Comparison — CHB-MIT External Validation Set")

# =============================================================================
# MCNEMAR
# =============================================================================
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar(pred_a, pred_b, y_true, label_a, label_b, dataset_name):
    b = int(((pred_a==y_true) & (pred_b!=y_true)).sum())
    c = int(((pred_a!=y_true) & (pred_b==y_true)).sum())
    if b+c == 0:
        print(f"  [{dataset_name}] {label_a} vs {label_b}: identical — skip"); return
    res = mcnemar([[0,b],[c,0]], exact=False, correction=True)
    print(f"  [{dataset_name}] {label_a} vs {label_b}: "
          f"chi2={res.statistic:.4f}  p={res.pvalue:.4f}")

rf_base_pred = next(r['y_pred'] for r in baseline_results       if r['Model']=='Random Forest')
cgan_rf_pred = next(r['y_pred'] for r in cgan_results           if 'Random Forest' in r['Model'])
rf_base_chb  = next(r['y_pred'] for r in baseline_chbmit_results if r['Model']=='Random Forest')
cgan_rf_chb  = next(r['y_pred'] for r in cgan_chbmit_results    if 'Random Forest' in r['Model'])

run_mcnemar(rf_base_pred, cgan_rf_pred, y_test,   'RF Baseline','RF CWGAN-GP','Bonn')
run_mcnemar(rf_base_chb,  cgan_rf_chb,  y_chbmit, 'RF Baseline','RF CWGAN-GP','CHB-MIT')

# =============================================================================
# AUGMENTATION RATIO STUDY
# =============================================================================
RATIOS = [0.0, 0.25, 0.50, 0.75, 1.00]
kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
ratio_rows = []

for ratio in RATIOS:
    p_list, r_list, f_list = [], [], []
    for tr_idx, val_idx in kf.split(X_train_scaled, y_train):
        X_tr_fold = X_train_scaled[tr_idx]; y_tr_fold = y_train[tr_idx]
        X_val     = X_train_scaled[val_idx]; y_val     = y_train[val_idx]
        if ratio > 0:
            n_synth = int(len(X_tr_fold) * ratio)
            X_s, y_s = generate_samples(n_synth, target_class=1)
            X_tr_fold = np.vstack([X_tr_fold, X_s])
            y_tr_fold = np.concatenate([y_tr_fold, y_s])
        clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
        clf.fit(X_tr_fold, y_tr_fold)
        yp = clf.predict(X_val)
        p, r, f, _ = precision_recall_fscore_support(
            y_val, yp, average='binary', pos_label=1, zero_division=0)
        p_list.append(p); r_list.append(r); f_list.append(f)
    ratio_rows.append({
        'Ratio': f'{int(ratio*100)}%',
        'Precision': np.mean(p_list), 'Recall': np.mean(r_list),
        'F1-Score': np.mean(f_list),
        'Prec_std': np.std(p_list),   'Rec_std': np.std(r_list),
    })

ratio_df = pd.DataFrame(ratio_rows)
print(ratio_df.to_string(index=False))

# =============================================================================
# DEEP LEARNING
# =============================================================================
from tensorflow.keras import layers as klayers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

EPOCHS   = 100
BATCH_DL = 64
TIMESTEPS  = 178
N_CLASSES  = 2
CLASS_WEIGHT_IMBALANCED = {0: 1.0, 1: 4.0}

def prepare_dl_data(X, y):
    return (X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32),
            to_categorical(y, num_classes=2).astype(np.float32))

X_train_dl,  y_train_dl  = prepare_dl_data(X_train_scaled, y_train)
X_test_dl,   y_test_dl   = prepare_dl_data(X_test_scaled,  y_test)
X_aug_dl,    y_aug_dl    = prepare_dl_data(X_aug_scaled,   y_aug)
X_chbmit_dl, y_chbmit_dl = prepare_dl_data(X_chbmit,       y_chbmit)

def get_callbacks():
    return [
        EarlyStopping(monitor='val_recall', patience=15,
                      restore_best_weights=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.AUC(name='auc')],
    )
    return model

def evaluate_dl_two_sets(name, model, X_te, y_te_int, X_te2, y_te2_int):
    def _eval(X, y_int):
        proba  = model.predict(X, verbose=0)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        TP = int(((y_pred==1)&(y_int==1)).sum()); FP = int(((y_pred==1)&(y_int==0)).sum())
        FN = int(((y_pred==0)&(y_int==1)).sum()); TN = int(((y_pred==0)&(y_int==0)).sum())
        prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
        rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1   = TP/(TP+0.5*FP+0.5*FN) if (TP+0.5*FP+0.5*FN)>0 else 0.0
        acc  = (TP+TN)/len(y_int)
        fpr_, tpr_, _ = roc_curve(y_int, proba)
        return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1,
                    AUC=auc(fpr_,tpr_), TP=TP, FP=FP, FN=FN, TN=TN,
                    proba=proba, y_pred=y_pred)

    r1 = _eval(X_te,  y_te_int)
    r2 = _eval(X_te2, y_te2_int)
    merged = {'Model': name}
    merged.update(r1)
    merged.update({k+'_chbmit': v for k, v in r2.items()})
    return merged

dl_results = []

def residual_block(x, filters, kernel_size=5, stride=1, downsample=False):
    shortcut = x
    x = klayers.Conv1D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = klayers.BatchNormalization()(x); x = klayers.ReLU()(x)
    x = klayers.Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = klayers.BatchNormalization()(x)
    if downsample or shortcut.shape[-1] != filters:
        shortcut = klayers.Conv1D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = klayers.BatchNormalization()(shortcut)
    x = klayers.Add()([x, shortcut]); x = klayers.ReLU()(x)
    return x

def build_resnet_1d():
    inp = Input(shape=(TIMESTEPS, 1))
    x   = klayers.Conv1D(32, 7, padding='same', use_bias=False)(inp)
    x   = klayers.BatchNormalization()(x); x = klayers.ReLU()(x)
    x   = residual_block(x, 32);  x = residual_block(x, 32)
    x   = residual_block(x, 64,  stride=2, downsample=True); x = residual_block(x, 64)
    x   = residual_block(x, 128, stride=2, downsample=True); x = residual_block(x, 128)
    x   = residual_block(x, 256, stride=2, downsample=True); x = residual_block(x, 256)
    x   = klayers.GlobalAveragePooling1D()(x); x = klayers.Dropout(0.4)(x)
    out = klayers.Dense(N_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='ResNet1D')

for label, X_tr, y_tr, cw in [
    ('Baseline', X_train_dl, y_train_dl, CLASS_WEIGHT_IMBALANCED),
    ('CWGAN-GP', X_aug_dl,   y_aug_dl,   None),
]:
    m = compile_model(build_resnet_1d())
    m.fit(X_tr, y_tr, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH_DL,
          class_weight=cw, callbacks=get_callbacks(), verbose=0)
    dl_results.append(evaluate_dl_two_sets(
        f'ResNet1D ({label})', m, X_test_dl, y_test, X_chbmit_dl, y_chbmit))

def build_cnn_lstm():
    inp = Input(shape=(TIMESTEPS, 1))
    x = klayers.Conv1D(32,  5, padding='same', activation='relu')(inp)
    x = klayers.BatchNormalization()(x); x = klayers.MaxPooling1D(2)(x)
    x = klayers.Conv1D(64,  5, padding='same', activation='relu')(x)
    x = klayers.BatchNormalization()(x); x = klayers.MaxPooling1D(2)(x)
    x = klayers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = klayers.BatchNormalization()(x); x = klayers.MaxPooling1D(2)(x)
    x = klayers.Bidirectional(klayers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    x = klayers.Bidirectional(klayers.LSTM(32, return_sequences=False, dropout=0.2))(x)
    x = klayers.Dense(64, activation='relu')(x); x = klayers.Dropout(0.4)(x)
    out = klayers.Dense(N_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='CNN_LSTM')

for label, X_tr, y_tr, cw in [
    ('Baseline', X_train_dl, y_train_dl, CLASS_WEIGHT_IMBALANCED),
    ('CWGAN-GP', X_aug_dl,   y_aug_dl,   None),
]:
    m = compile_model(build_cnn_lstm())
    m.fit(X_tr, y_tr, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH_DL,
          class_weight=cw, callbacks=get_callbacks(), verbose=0)
    dl_results.append(evaluate_dl_two_sets(
        f'CNN-LSTM ({label})', m, X_test_dl, y_test, X_chbmit_dl, y_chbmit))

def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    dims      = np.arange(depth)[np.newaxis, :]
    angles    = positions / np.power(10000, (2*(dims//2))/depth)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

class TransformerBlock(klayers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = klayers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn   = keras.Sequential([klayers.Dense(ff_dim, activation='relu'), klayers.Dense(embed_dim)])
        self.norm1 = klayers.LayerNormalization(epsilon=1e-6)
        self.norm2 = klayers.LayerNormalization(epsilon=1e-6)
        self.drop1 = klayers.Dropout(dropout_rate)
        self.drop2 = klayers.Dropout(dropout_rate)

    def call(self, x, training=False):
        out = self.drop1(self.attn(x, x, training=training), training=training)
        x   = self.norm1(x + out)
        out = self.drop2(self.ffn(x), training=training)
        return self.norm2(x + out)

def build_transformer_eeg(embed_dim=64, num_heads=4, ff_dim=128, num_blocks=3):
    pos_enc_np = positional_encoding(TIMESTEPS, embed_dim).numpy()
    inp = Input(shape=(TIMESTEPS, 1))
    x   = klayers.Dense(embed_dim)(inp)
    x   = klayers.Lambda(lambda t: t + pos_enc_np)(x)
    x   = klayers.Dropout(0.1)(x)
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, 0.1)(x)
    x   = klayers.GlobalAveragePooling1D()(x); x = klayers.Dropout(0.3)(x)
    x   = klayers.Dense(64, activation='relu')(x); x = klayers.Dropout(0.3)(x)
    out = klayers.Dense(N_CLASSES, activation='softmax')(x)
    return Model(inp, out, name='Transformer_EEG')

for label, X_tr, y_tr, cw in [
    ('Baseline', X_train_dl, y_train_dl, CLASS_WEIGHT_IMBALANCED),
    ('CWGAN-GP', X_aug_dl,   y_aug_dl,   None),
]:
    m = compile_model(build_transformer_eeg())
    m.fit(X_tr, y_tr, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH_DL,
          class_weight=cw, callbacks=get_callbacks(), verbose=0)
    dl_results.append(evaluate_dl_two_sets(
        f'Transformer ({label})', m, X_test_dl, y_test, X_chbmit_dl, y_chbmit))

# ── Extra McNemar comparisons (DL vs. RF baseline) -- referenced in the
#    paper appendix but not previously computed by this codebase; only
#    needs predictions already computed above, no retraining required.
print("\nExtra McNemar comparisons (DL vs. RF baseline):")
for dl_r in dl_results:
    run_mcnemar(rf_base_pred, dl_r['y_pred'], y_test,
                'RF Baseline', dl_r['Model'], 'Bonn')
    run_mcnemar(rf_base_chb, dl_r['y_pred_chbmit'], y_chbmit,
                'RF Baseline', dl_r['Model'], 'CHB-MIT')

# =============================================================================
# FIGURE GENERATION (all remaining plots -- runs automatically, no separate step)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ── Helper functions (reconstructed — same as original pipeline) ───────────
def plot_confusion_matrix(name, y_te, y_pred, filename=None):
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Non-seizure','Seizure'],
                yticklabels=['Non-seizure','Seizure'], ax=ax)
    ax.set_title(name); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_from_model(name, model, X_te, y_te, filename=None):
    if not hasattr(model, 'predict_proba'):
        return
    proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC — {name}'); plt.legend(); plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_from_proba(name, proba, y_te, filename=None):
    fpr, tpr, _ = roc_curve(y_te, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC — {name}'); plt.legend(); plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

print("Generating remaining figures...")

# ── 1. GAN loss curves ───────────────────────────────────────────────────────
steps_arr = list(range(len(critic_losses)))
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(steps_arr, generator_losses, linewidth=0.8, color='#2e86c1')
axes[0].set_ylabel('Generator Loss')
axes[0].set_title('CWGAN-GP Training Dynamics')
axes[1].plot(steps_arr, critic_losses, linewidth=0.8, color='#e74c3c')
axes[1].set_ylabel('Critic Loss (Wasserstein Estimate)')
axes[1].set_xlabel('Training Step')
plt.tight_layout()
plt.savefig('fig_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_loss_curves.png")

# ── 2. Real vs synthetic ictal waveforms ────────────────────────────────────
synth_display_raw = scaler.inverse_transform(X_synth_scaled[:4])
real_display_raw  = X_test[:4]
fig, axes = plt.subplots(2, 4, figsize=(16, 5))
for i in range(4):
    axes[0,i].plot(real_display_raw[i],  linewidth=0.8, color='#2e86c1')
    axes[0,i].set_title(f'Real Ictal {i+1}', fontsize=9)
    axes[1,i].plot(synth_display_raw[i], linewidth=0.8, color='#e74c3c')
    axes[1,i].set_title(f'Synthetic Ictal {i+1}', fontsize=9)
    for row in range(2):
        axes[row,i].set_xlabel('Time step')
        if i == 0: axes[row,i].set_ylabel('Amplitude (μV)')
plt.suptitle('Real vs CWGAN-GP Generated Ictal EEG', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_real_vs_synthetic.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_real_vs_synthetic.png")

# ── 3. Latent space interpolation ───────────────────────────────────────────
z_start = np.random.normal(0, 1, (1, NOISE_DIM))
z_end   = np.random.normal(0, 1, (1, NOISE_DIM))
plt.figure(figsize=(13, 7))
for i, alpha in enumerate(np.linspace(0, 1, 10)):
    z_interp   = (1-alpha)*z_start + alpha*z_end
    lbl        = tf.one_hot([1], depth=NUM_CLASSES)
    sig_scaled = ema_generator.predict([z_interp, lbl], verbose=0)
    sig        = scaler.inverse_transform(sig_scaled).flatten()
    plt.plot(sig + i*1000, label=f'Step {i}', linewidth=0.9)
plt.title('Latent Space Interpolation: Ictal Morphology Evolution', fontweight='bold')
plt.xlabel('Time step'); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig('fig_latent_interpolation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_latent_interpolation.png")

# ── 4. PSD comparison: Bonn real / synthetic / CHB-MIT ──────────────────────
chbmit_ictal_raw = X_chbmit_raw[y_chbmit == 1]
f_real,  p_real  = welch(real_ictal_all_raw, fs=178, axis=1)
f_synth, p_synth = welch(synth_eval_raw,     fs=178, axis=1)
f_chb,   p_chb   = welch(chbmit_ictal_raw,   fs=178, axis=1)
plt.figure(figsize=(10, 5))
plt.semilogy(f_real,  np.mean(p_real,  axis=0), label='Real Ictal (Bonn)',    color='blue',   linewidth=2)
plt.semilogy(f_synth, np.mean(p_synth, axis=0), label='Synthetic Ictal',      color='orange', linewidth=2, linestyle='--')
plt.semilogy(f_chb,   np.mean(p_chb,   axis=0), label='Real Ictal (CHB-MIT)', color='green',  linewidth=2, linestyle=':')
plt.title('PSD: Bonn vs CWGAN-GP Synthetic vs CHB-MIT Ictal EEG', fontweight='bold')
plt.xlabel('Frequency (Hz)'); plt.ylabel('Power (μV²/Hz)')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_psd_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_psd_comparison.png")

# ── 5. Augmentation ratio plot ───────────────────────────────────────────────
x_vals = [0, 25, 50, 75, 100]
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x_vals, ratio_df['Precision'], yerr=ratio_df['Prec_std'],
            fmt='o-', color='#2e86c1', linewidth=2, capsize=4, label='Precision')
ax.errorbar(x_vals, ratio_df['Recall'],    yerr=ratio_df['Rec_std'],
            fmt='s-', color='#e74c3c', linewidth=2, capsize=4, label='Recall')
ax.plot(x_vals, ratio_df['F1-Score'], 'g^-', linewidth=2, label='F1-Score')
ax.set_xlabel('Augmentation Ratio (% of training set)'); ax.set_ylabel('Score')
ax.set_title('RF Performance vs Augmentation Ratio (5-fold CV, Bonn)', fontweight='bold')
ax.set_xticks(x_vals); ax.set_xticklabels([f'{r}%' for r in x_vals])
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('fig_augmentation_ratio.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_augmentation_ratio.png")

# ── 6. UMAP (optional — needs umap-learn) ───────────────────────────────────
try:
    import umap
    bonn_for_umap     = real_ictal_all_scaled[:300]
    synth_for_umap, _ = generate_samples(300, target_class=1)
    chbmit_for_umap   = X_chbmit[y_chbmit == 1][:300]
    combined  = np.vstack([bonn_for_umap, synth_for_umap, chbmit_for_umap])
    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=SEED)
    embedding = reducer.fit_transform(combined)
    n1, n2, n3 = len(bonn_for_umap), len(synth_for_umap), len(chbmit_for_umap)
    plt.figure(figsize=(9, 7))
    plt.scatter(embedding[:n1,0], embedding[:n1,1], alpha=0.5, s=15,
                label='Real Ictal (Bonn)', color='#2e86c1')
    plt.scatter(embedding[n1:n1+n2,0], embedding[n1:n1+n2,1], alpha=0.35, s=15,
                label='Synthetic Ictal', color='#e74c3c')
    plt.scatter(embedding[n1+n2:,0], embedding[n1+n2:,1], alpha=0.5, s=15,
                label='Real Ictal (CHB-MIT)', color='#27ae60')
    plt.title('UMAP: Bonn vs CWGAN-GP Synthetic vs CHB-MIT Ictal EEG', fontweight='bold')
    plt.xlabel('UMAP Dim 1'); plt.ylabel('UMAP Dim 2'); plt.legend()
    plt.tight_layout()
    plt.savefig('fig_umap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  fig_umap.png")
except ImportError:
    print("  [skipped] pip install umap-learn to enable fig_umap.png")

# ── 7. SHAP (optional — needs shap) ─────────────────────────────────────────
try:
    import shap
    rf_aug    = aug_models['Random Forest']
    explainer = shap.TreeExplainer(rf_aug)
    X_shap    = np.vstack([X_test_scaled[:150], X_chbmit[:50]])
    shap_vals = explainer.shap_values(X_shap)
    shap_ictal = (shap_vals[1] if isinstance(shap_vals, list)
                  else np.array(shap_vals)[:,:,1] if len(np.array(shap_vals).shape)==3
                  else shap_vals)
    global_imp = np.abs(shap_ictal).mean(axis=0)
    top_idx    = np.argsort(global_imp)[::-1][:20]
    top_vals   = global_imp[top_idx]
    top_labels = [f'X{i+1}  (t≈{i+1}ms)' for i in top_idx]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(20), top_vals[::-1],
            color=['#e74c3c' if v > top_vals[2] else '#2e86c1' for v in top_vals[::-1]])
    ax.set_yticks(range(20)); ax.set_yticklabels(top_labels[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Feature Importance — CWGAN-GP RF', fontweight='bold')
    ax.grid(axis='x', alpha=0.3); plt.tight_layout()
    plt.savefig('fig_shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  fig_shap_importance.png")
    print("\n  Top 5 features:")
    for rank, (idx, val) in enumerate(zip(top_idx[:5], top_vals[:5]), 1):
        print(f"    {rank}. X{idx+1} (t≈{idx+1}ms)  mean|SHAP|={val:.5f}")
except ImportError:
    print("  [skipped] pip install shap to enable fig_shap_importance.png")

# ── 8. Confusion matrices + ROC curves for classical models ────────────────
for r in baseline_results:
    plot_confusion_matrix(f"Baseline (Bonn): {r['Model']}", y_test, r['y_pred'],
                          f"fig_cm_baseline_{r['Model'].replace(' ','_')}.png")
    plot_roc_from_model(f"Baseline (Bonn): {r['Model']}", r['model'], X_test_scaled, y_test,
             f"fig_roc_baseline_{r['Model'].replace(' ','_')}.png")
for r_chb, r_bonn in zip(baseline_chbmit_results, baseline_results):
    plot_confusion_matrix(f"Baseline (CHB-MIT): {r_chb['Model']}", y_chbmit, r_chb['y_pred'],
                          f"fig_cm_baseline_chbmit_{r_chb['Model'].replace(' ','_')}.png")
    plot_roc_from_model(f"Baseline (CHB-MIT): {r_chb['Model']}", r_bonn['model'], X_chbmit, y_chbmit,
             f"fig_roc_baseline_chbmit_{r_chb['Model'].replace(' ','_')}.png")
for r_bonn, r_chb in zip(cgan_results, cgan_chbmit_results):
    plot_confusion_matrix(f"{r_bonn['Model']} [Bonn]", y_test, r_bonn['y_pred'],
                          f"fig_cm_{r_bonn['Model'].replace(' ','_')}_bonn.png")
    plot_confusion_matrix(f"{r_chb['Model']} [CHB-MIT]", y_chbmit, r_chb['y_pred'],
                          f"fig_cm_{r_chb['Model'].replace(' ','_')}_chbmit.png")
    plot_roc_from_model(f"{r_bonn['Model']} [Bonn]", r_bonn['model'], X_test_scaled, y_test,
             f"fig_roc_{r_bonn['Model'].replace(' ','_')}_bonn.png")
    plot_roc_from_model(f"{r_chb['Model']} [CHB-MIT]", r_bonn['model'], X_chbmit, y_chbmit,
             f"fig_roc_{r_chb['Model'].replace(' ','_')}_chbmit.png")
print("  confusion matrices + ROC curves for classical models")

# ── 9. ROC curves for deep learning models (Bonn + CHB-MIT) ────────────────
plt.figure(figsize=(6,5))
for r in dl_results:
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC — Deep Learning (Bonn)')
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig('fig_roc_dl_bonn.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(6,5))
for r in dl_results:
    fpr, tpr, _ = roc_curve(y_chbmit, r['proba_chbmit'])
    plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC_chbmit']:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC — Deep Learning (CHB-MIT)')
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig('fig_roc_dl_chbmit.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_roc_dl_bonn.png, fig_roc_dl_chbmit.png")

# ── 10. Print full DL results table (Precision/F1/AUC weren't printed earlier) ─
print("\n" + "="*70)
print("FULL DEEP LEARNING RESULTS (for the paper's Table 7/8)")
print("="*70)
for r in dl_results:
    print(f"{r['Model']:20s} Bonn:    Acc={r['Accuracy']:.3f} Prec={r['Precision']:.3f} "
          f"Rec={r['Recall']:.3f} F1={r['F1']:.3f} AUC={r['AUC']:.3f}")
    print(f"{'':20s} CHB-MIT: Acc={r['Accuracy_chbmit']:.3f} Prec={r['Precision_chbmit']:.3f} "
          f"Rec={r['Recall_chbmit']:.3f} F1={r['F1_chbmit']:.3f} AUC={r['AUC_chbmit']:.3f}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("COMPLETE RESULTS SUMMARY")
print("=" * 60)
print(f"\n  GAN Quality Metrics: JS={js_div:.4f}  MMD={mmd:.6f}  Vendi={vs:.2f}")
for r in dl_results:
    print(f"    {r['Model']:45s} Bonn Recall={r['Recall']:.3f}  "
          f"CHB-MIT Recall={r['Recall_chbmit']:.3f}")
print("\nAll outputs saved.")
print("=" * 60)


# =============================================================================
# COPY ALL OUTPUTS TO DRIVE (Colab-safe -- no-ops if not running in Colab /
# Drive isn't mounted, so this is also safe to run locally/non-Colab)
# =============================================================================
import shutil, glob
try:
    drive_out = os.environ.get("EEG_RESULTS_DIR", "/content/drive/MyDrive/eeg_project/results")
    if os.path.isdir(os.path.dirname(drive_out)) or "/content/drive" in drive_out:
        os.makedirs(drive_out, exist_ok=True)
        copied = 0
        for pattern in ("fig_*.png", "*.pkl"):
            for f in glob.glob(pattern):
                shutil.copy(f, drive_out)
                copied += 1
        print(f"\nCopied {copied} files to {drive_out}")
    else:
        print(f"\n[Skipped Drive copy] {drive_out} not available in this environment; "
              f"figures and scaler remain in the current working directory.")
except Exception as e:
    print(f"\n[Skipped Drive copy] {e}")