import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
# 1) 데이터 로딩
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
        samples.append(seq.reshape(-1, 1))
    return np.stack(samples, axis=0)

data = load_dataset(ROOT_DIR, SUB_FOLDERS)
print("loaded :", data.shape)

# ─────────────────────────────────────────
# 2) 배치‑제너레이터
# ─────────────────────────────────────────
def seq_gen(full_seq, batch=4, n_min=20, n_max=400):
    B, _, _ = full_seq.shape
    while True:
        ns    = np.random.randint(n_min, n_max+1, size=batch)
        max_n = ns.max()
        T_out = TOTAL_HOURS - max_n
        xs, ys = [], []
        for n in ns:
            idx  = np.random.randint(0, B)
            seq  = full_seq[idx]
            x    = seq[:n]
            y    = seq[n:n+T_out]
            xs.append(np.pad(x, ((0,max_n-n),(0,0))))
            ys.append(y)
        yield np.stack(xs, 0), np.stack(ys, 0)

# ─────────────────────────────────────────
# 3) RepeatVectorDynamic
# ─────────────────────────────────────────
class RepeatVectorDynamic(Layer):
    def call(self, inputs):
        h, enc_in = inputs
        t_out = TOTAL_HOURS - tf.shape(enc_in)[1]
        return tf.repeat(h[:, tf.newaxis, :], t_out, axis=1)
    def compute_output_shape(self, s):
        return (s[0][0], None, s[0][-1])

# ─────────────────────────────────────────
# 4) 모델 정의: 학습용(train) vs 추론용(inference)
# ─────────────────────────────────────────
def build_models():
    enc_in = Input(shape=(None,1), name="enc_in")
    x = Masking(0.0)(enc_in)

    # Encoder
    for i in range(N_LSTM_LAYERS):
        if i == N_LSTM_LAYERS-1:
            x, h, c = LSTM(N_LSTM_UNITS, return_sequences=True, return_state=True,
                           dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
                           name=f'enc_lstm_{i+1}')(x)
        else:
            x = LSTM(N_LSTM_UNITS, return_sequences=True,
                     dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
                     name=f'enc_lstm_{i+1}')(x)
    enc_seq = x

    rep = RepeatVectorDynamic()([h, enc_in])
    dec = rep
    for i in range(N_LSTM_LAYERS):
        dec = LSTM(N_LSTM_UNITS, return_sequences=True,
                   dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
                   name=f'dec_lstm_{i+1}')(dec,
                   initial_state=[h, c] if i==0 else None)

    # Attention layer
    att_layer = AdditiveAttention(name="attention")
    att, att_weights = att_layer([dec, enc_seq, enc_seq], mask=[None, None], return_attention_scores=True)

    cat = Concatenate()([dec, att])
    mid = TimeDistributed(Dense(N_LSTM_UNITS, activation='tanh'))(cat)
    mid = LayerNormalization()(mid)
    mid = Dropout(DROPOUT_RATE)(mid)
    out = TimeDistributed(Dense(1), name="output")(mid)

    # 학습용 모델: predictions만 출력
    train_model = Model(enc_in, out, name="Seq2Seq_Train")
    train_model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='mse')

    # 추론용 모델: 예측값 + attention 가중치 출력
    infer_model = Model(enc_in, [out, att_weights], name="Seq2Seq_Infer")

    return train_model, infer_model

train_model, infer_model = build_models()
train_model.summary()
infer_model.summary()

# ─────────────────────────────────────────
# 5) 콜백 정의
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=PATIENCE_EARLY_STOP, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=PATIENCE_LR_SCHED, factor=0.5, verbose=1)
]

# ─────────────────────────────────────────
# 6) 학습 및 저장
# ─────────────────────────────────────────
train_gen = seq_gen(data, batch=BATCH_SIZE, n_min=N_MIN, n_max=N_MAX)
val_gen   = seq_gen(data, batch=BATCH_SIZE, n_min=N_MIN+50, n_max=N_MAX+50)

train_model.fit(
    train_gen,
    epochs=30,
    steps_per_epoch=20,
    validation_data=val_gen,
    validation_steps=20,
    callbacks=callbacks,
    verbose=1
)
train_model.save("final_model.h5")

# ─────────────────────────────────────────
# 7) 예측·시각화·XAI 분석 함수
# ─────────────────────────────────────────
def predict_plot_and_explain(model: Model, input_seq: np.ndarray, n: int):
    preds, att_w = model.predict(input_seq[np.newaxis,...])
    preds = preds[0,:,0]
    att_w = att_w[0]

    plt.figure()
    plt.plot(range(n), input_seq[:,0], label='Input')
    plt.plot(range(n, n+len(preds)), preds, label='Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Freshness Value')
    plt.legend()
    plt.show()

    plt.figure()
    plt.imshow(att_w.T, aspect='auto')
    plt.xlabel('Output Steps')
    plt.ylabel('Input Steps')
    plt.title('Attention Weights Heatmap')
    plt.colorbar()
    plt.show()

    return preds, att_w

# ─────────────────────────────────────────
# 8) 사용자 입력 처리 및 실행 예시
# ─────────────────────────────────────────
if __name__ == "__main__":
    n_input = int(input("Enter number of input hours (n): "))
    print(f"Please enter {n_input} freshness values, separated by spaces:")
    values = list(map(float, input().split()))
    if len(values) != n_input:
        raise ValueError(f"Expected {n_input} values, got {len(values)}")
    user_seq = np.array(values, dtype=np.float32).reshape(-1,1)
    predictions, attention_weights = predict_plot_and_explain(infer_model, user_seq, n_input)
