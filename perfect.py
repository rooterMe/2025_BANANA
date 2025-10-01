# perfect_v2.py  ── TOTAL_HOURS = 736  버전
import numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import (
    Input, Masking, LSTM, Dense, Dropout, LayerNormalization,
    AdditiveAttention, Concatenate, TimeDistributed, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────
# 0) 상수 · 하이퍼파라미터
# ─────────────────────────────────────────
TOTAL_HOURS          = 736
ROOT_DIR             = Path(r"C:\Users\parkw\Downloads\Dataset")
SUB_FOLDERS          = [f"banana_{i}" for i in range(5)]
N_LSTM_UNITS         = 128
N_LSTM_LAYERS        = 2
DROPOUT_RATE         = 0.2
RECURRENT_DROPOUT    = 0.2
LEARNING_RATE        = 1e-3
PATIENCE_EARLY_STOP  = 10
PATIENCE_LR_SCHED    = 5
BATCH_SIZE           = 4
N_MIN, N_MAX         = 20, 400

# ─────────────────────────────────────────
# 1) 데이터 로딩 함수
# ─────────────────────────────────────────
def load_dataset(root: Path, folders: list[str]) -> np.ndarray:
    samples = []
    for fd in folders:
        label_dir = root / fd / "label"
        txts = sorted(label_dir.glob("*.txt"), key=lambda p: int(p.stem))
        seq = np.zeros(TOTAL_HOURS, dtype=np.float32)
        for p in txts:
            with open(p) as f:
                t, v = f.read().strip().splitlines()
            t, v = int(float(t)), float(v)
            if 0 <= t < TOTAL_HOURS:
                seq[t] = v
        samples.append(seq.reshape(-1, 1))           # (736,1)
    return np.stack(samples, axis=0)                 # (B,736,1)

data = load_dataset(ROOT_DIR, SUB_FOLDERS)
print("loaded :", data.shape)

# ─────────────────────────────────────────
# 2) 배치‑제너레이터
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# 2) 배치‑제너레이터 (shape mismatch fix)
# ─────────────────────────────────────────
def seq_gen(full_seq, batch=4, n_min=20, n_max=400):
    B, _, _ = full_seq.shape
    while True:
        ns      = np.random.randint(n_min, n_max+1, size=batch)
        max_n   = ns.max()
        T_out   = TOTAL_HOURS - max_n          # ← 모델 출력 길이
        xs, ys  = [], []
        for n in ns:
            idx  = np.random.randint(0, B)
            seq  = full_seq[idx]               # (736,1)
            x    = seq[:n]                     # (n,1)
            y    = seq[n:n+T_out]              # 잘라서 길이 T_out 로 맞춤
            # 만약 n 이 max_n 과 같아 y 길이가 정확히 T_out,
            # 작으면 항상 T_out 보다 김 → 슬라이스 충분
            xs.append(np.pad(x, ((0, max_n-n),(0,0))))   # (max_n,1)
            ys.append(y)                                 # (T_out,1)
        yield np.stack(xs, 0), np.stack(ys, 0)

# ─────────────────────────────────────────
# 3) RepeatVectorDynamic (리팩터링)  ### <-- MOD
# ─────────────────────────────────────────
# ─────────────────────────────────────────
class RepeatVectorDynamic(Layer):
    """컨텍스트 h 를 (736‑n) 길이로 복제"""
    def call(self, inputs):
        h, enc_in = inputs                 # h:(B,U)  enc_in:(B,n,1)
        t_out = TOTAL_HOURS - tf.shape(enc_in)[1]  # 736‑n  (동적)
        # h → (B,1,U)  → (B,t_out,U)
        return tf.repeat(h[:, tf.newaxis, :], t_out, axis=1)

    # output‑shape 추론만 남겨 둬도 무방
    def compute_output_shape(self, s):
        return (s[0][0], None, s[0][-1])
# ─────────────────────────────────────────
# 4) 모델 빌드 함수 (Encoder·Decoder 전체)
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# 4) 모델 빌드 함수 (Encoder·Decoder 전체)
# ─────────────────────────────────────────
def build_model() -> Model:
    enc_in = Input(shape=(None,1), name="enc_in")
    x      = Masking(0.0)(enc_in)

    # ---------- Encoder ----------
    for i in range(N_LSTM_LAYERS):
        if i == N_LSTM_LAYERS - 1:                # ★ 마지막 인코더 층
            x, h, c = LSTM(
                N_LSTM_UNITS,
                return_sequences=True,            # ← 반드시 True  (수정)  ※
                return_state=True,
                dropout=DROPOUT_RATE,
                recurrent_dropout=RECURRENT_DROPOUT,
                name=f'enc_lstm_{i+1}'
            )(x)
        else:                                     # 중간 층
            x = LSTM(
                N_LSTM_UNITS,
                return_sequences=True,
                dropout=DROPOUT_RATE,
                recurrent_dropout=RECURRENT_DROPOUT,
                name=f'enc_lstm_{i+1}'
            )(x)
    enc_seq = x                                   # (B, n, U)  ← 3‑D 유지

    # ----- 이하 Decoder, Attention, Compile 부분은 동일 -----


    # Repeat h -> (736-n)
    rep = RepeatVectorDynamic()([h, enc_in])         # _keras_mask 제거 ### <-- MOD

    # Decoder
    dec = rep
    for i in range(N_LSTM_LAYERS):
        dec = LSTM(N_LSTM_UNITS, return_sequences=True,
                   dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
                   name=f'dec_lstm_{i+1}')(dec, initial_state=[h, c] if i == 0 else None)

    # Attention + 출력
# ----- Decoder까지 동일 ...

# ----- Attention & 후처리 (★ 수정)
# ... Encoder & Decoder 부분 동일 ...

# ----- Attention + 후처리 (마스크 끄기)
    att = AdditiveAttention(name="attention")(
            [dec, enc_seq, enc_seq],    # (query, key, value)
            mask=[None, None])          # ★ 마스크 비활성

    cat = Concatenate()([dec, att])
    mid = TimeDistributed(Dense(N_LSTM_UNITS, activation='tanh'))(cat)
    mid = LayerNormalization()(mid)
    mid = Dropout(DROPOUT_RATE)(mid)
    out = TimeDistributed(Dense(1))(mid)



    m = Model(enc_in, out, name="Seq2Seq_736")
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='mse')
    return m

model = build_model()
model.summary()

# ─────────────────────────────────────────
# 5) 학습
# ─────────────────────────────────────────
train_gen = seq_gen(data, batch=BATCH_SIZE, n_min=N_MIN, n_max=N_MAX)
val_gen   = seq_gen(data, batch=BATCH_SIZE, n_min=N_MIN+50, n_max=N_MAX+50)

callbacks = [
    EarlyStopping(patience=PATIENCE_EARLY_STOP,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=PATIENCE_LR_SCHED,
                      factor=0.5, verbose=1)
]

model.fit(train_gen,
          epochs=60, steps_per_epoch=200,
          validation_data=val_gen,
          validation_steps=40,
          callbacks=callbacks,
          verbose=1)

# ─────────────────────────────────────────
# 6) 예측 예시
# ─────────────────────────────────────────
sample = data[0, :30]                      # 앞 30 시간
pred   = model.predict(sample[np.newaxis,...])
print("예측 shape :", pred.shape)          # (1, 706, 1) = 736 - 30
