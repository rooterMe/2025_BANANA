import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, AdditiveAttention, Concatenate,
    TimeDistributed, Dense, Dropout, LayerNormalization, Layer
)
from tensorflow.keras.models import Model

# ─────────────────────────────────────────
# 하이퍼파라미터 · 상수
# ─────────────────────────────────────────
TOTAL_HOURS       = 736
N_LSTM_UNITS      = 128
N_LSTM_LAYERS     = 2
DROPOUT_RATE      = 0.2
RECURRENT_DROPOUT = 0.2

# ─────────────────────────────────────────
# Custom Layer: RepeatVectorDynamic (추론용)
# ─────────────────────────────────────────
class RepeatVectorDynamic(Layer):
    def call(self, inputs):
        h, enc_in = inputs
        t_out = TOTAL_HOURS - tf.shape(enc_in)[1]
        return tf.repeat(h[:, tf.newaxis, :], t_out, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], None, input_shape[0][-1])

# ─────────────────────────────────────────
# 추론 모델 빌드
# ─────────────────────────────────────────
def build_inference_model():
    enc_in = Input(shape=(None,1), name="enc_in")
    x = enc_in

    # Encoder
    for i in range(N_LSTM_LAYERS):
        if i == N_LSTM_LAYERS - 1:
            x, h, c = LSTM(
                N_LSTM_UNITS, return_sequences=True, return_state=True,
                dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
                name=f'enc_lstm_{i+1}'
            )(x)
        else:
            x = LSTM(
                N_LSTM_UNITS, return_sequences=True,
                dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
                name=f'enc_lstm_{i+1}'
            )(x)
    enc_seq = x

    # Repeat h for decoder input
    rep = RepeatVectorDynamic()([h, enc_in])

    # Decoder
    dec = rep
    for i in range(N_LSTM_LAYERS):
        dec = LSTM(
            N_LSTM_UNITS, return_sequences=True,
            dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
            name=f'dec_lstm_{i+1}'
        )(dec, initial_state=[h, c] if i == 0 else None)

    # Attention
    att_layer = AdditiveAttention(name="attention")
    context, att_w = att_layer([dec, enc_seq, enc_seq], return_attention_scores=True)
    cat = Concatenate()([dec, context])
    mid = TimeDistributed(Dense(N_LSTM_UNITS, activation='tanh'))(cat)
    mid = LayerNormalization()(mid)
    mid = Dropout(DROPOUT_RATE)(mid)
    out = TimeDistributed(Dense(1), name="output")(mid)

    return Model(enc_in, [out, att_w], name="Seq2Seq_Infer")

# ─────────────────────────────────────────
# 메인: 모델 로드 · 예측 · 시각화
# ─────────────────────────────────────────
def main():
    # 모델 아키텍처 생성 후 가중치 로드
    model = build_inference_model()
    model.load_weights('final_model.h5', by_name=True)

    # 사용자 입력
    n_input = int(input("Enter number of input hours (n): "))
    print(f"Please enter {n_input} freshness values, separated by spaces:")
    values = list(map(float, input().split()))
    if len(values) != n_input:
        raise ValueError(f"Expected {n_input} values, got {len(values)}")

    # 입력 배열 생성
    user_seq = np.array(values, dtype=np.float32).reshape(1, n_input, 1)

    # 예측 및 Attention 가중치
    preds, att_w = model.predict(user_seq)
    preds = preds[0, :, 0]
    att_w = att_w[0]

    # 예측 결과 시각화
    plt.figure()
    plt.plot(range(n_input), values, label='Input')
    plt.plot(range(n_input, n_input + len(preds)), preds, label='Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Freshness Value')
    plt.legend()
    plt.title('Input vs. Predicted Freshness')
    plt.show()

    # Attention Heatmap 시각화
    plt.figure()
    plt.imshow(att_w.T, aspect='auto')
    plt.xlabel('Output Steps')
    plt.ylabel('Input Steps')
    plt.title('Attention Weights Heatmap')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
