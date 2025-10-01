# ─── visual_core.py ───────────────────────────────────────────────
import os
# (1) Graphviz 실행파일 경로(설치 위치에 맞게 수정 가능)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

import tensorflow as tf
from tensorflow.keras.utils import plot_model

# ────────────────────────────────────────────────────────────────
# 2) 커스텀 레이어 & 모델 빌드 함수
# ────────────────────────────────────────────────────────────────
TOTAL_HOURS = 400

class RepeatVectorDynamic(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        final_h, encoder_in = inputs
        in_seq_len = tf.shape(encoder_in)[1]
        pred_len   = TOTAL_HOURS - in_seq_len
        return tf.tile(tf.expand_dims(final_h, 1), [1, pred_len, 1])

def build_seq2seq_flexible(
    N_LSTM_UNITS=128,
    N_LSTM_LAYERS=2,
    DROPOUT_RATE=0.2,
    RECURRENT_DROPOUT_RATE=0.2
):
    # (인코더)
    enc_in  = tf.keras.Input(shape=(None, 1), name='encoder_input')
    x       = enc_in
    states  = None
    for i in range(N_LSTM_LAYERS):
        x, h, c = tf.keras.layers.LSTM(
            N_LSTM_UNITS,
            return_sequences=True,
            return_state=True,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT_RATE,
            name=f'encoder_lstm_{i+1}'
        )(x)
        if i == N_LSTM_LAYERS - 1:
            states = [h, c]

    # (디코더)
    y = RepeatVectorDynamic(name='dynamic_repeat')([states[0], enc_in])
    for i in range(N_LSTM_LAYERS):
        y = tf.keras.layers.LSTM(
            N_LSTM_UNITS,
            return_sequences=True,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT_RATE,
            name=f'decoder_lstm_{i+1}'
        )(y if i else y, initial_state=states if i == 0 else None)

    # (어텐션 + 후처리)
    a       = tf.keras.layers.AdditiveAttention(name='attention')([y, x])
    cat     = tf.keras.layers.Concatenate()([y, a])
    td1     = tf.keras.layers.TimeDistributed(
                 tf.keras.layers.Dense(N_LSTM_UNITS, activation='tanh')
             )(cat)
    ln      = tf.keras.layers.LayerNormalization()(td1)
    dp      = tf.keras.layers.Dropout(DROPOUT_RATE)(ln)
    out     = tf.keras.layers.TimeDistributed(
                 tf.keras.layers.Dense(1), name='output')(dp)

    return tf.keras.Model(enc_in, out, name='seq2seq_full')

# ────────────────────────────────────────────────────────────────
# 3) 전체 모델 생성 + 가중치 로드
# ────────────────────────────────────────────────────────────────
full_model = build_seq2seq_flexible()
full_model.load_weights("final_model1.h5")       # 위치 맞춰서 수정 가능

# ────────────────────────────────────────────────────────────────
# 4) attention 층까지만 남긴 서브모델
# ────────────────────────────────────────────────────────────────
viz_model = tf.keras.Model(
    inputs  = full_model.input,
    outputs = full_model.get_layer('attention').output,
    name    = 'encoder_decoder_attention'
)

# ────────────────────────────────────────────────────────────────
# 5) PNG 시각화
# ────────────────────────────────────────────────────────────────
save_path = r"C:\Users\parkw\Downloads\seq2seq_core.png"   # 저장 경로
plot_model(
    viz_model,
    to_file      = save_path,
    show_shapes  = False,
    show_dtype   = False,
    expand_nested= False,
    rankdir      = 'LR',   # 왼쪽→오른쪽
    dpi          = 120
)
print(f"✅ 그래프 저장 완료 → {save_path}")
# ────────────────────────────────────────────────────────────────
