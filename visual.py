# visual.py

import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
import tensorflow as tf
from tensorflow.keras.utils import plot_model, model_to_dot

########################################################################
# 1) 커스텀 레이어 & build_seq2seq_flexible 함수 정의
########################################################################
TOTAL_HOURS = 400
class RepeatVectorDynamic(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        final_h, encoder_in = inputs
        in_seq_len = tf.shape(encoder_in)[1]
        pred_len   = TOTAL_HOURS - in_seq_len
        return tf.tile(tf.expand_dims(final_h, 1), [1, pred_len, 1])
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], None, input_shape[0][-1])

def build_seq2seq_flexible(
    N_LSTM_UNITS=128,
    N_LSTM_LAYERS=2,
    DROPOUT_RATE=0.2,
    RECURRENT_DROPOUT_RATE=0.2
):
    # (인코더)
    encoder_inputs = tf.keras.Input(shape=(None,1), name='encoder_input')
    x = encoder_inputs
    states = None
    for i in range(N_LSTM_LAYERS):
        lstm = tf.keras.layers.LSTM(
            N_LSTM_UNITS,
            return_sequences=True,
            return_state=True,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT_RATE,
            name=f'encoder_lstm_{i+1}'
        )
        x, h, c = lstm(x)
        if i == N_LSTM_LAYERS-1:
            states = [h, c]
    # (디코더)
    y = RepeatVectorDynamic(name='dynamic_repeat')([states[0], encoder_inputs])
    for i in range(N_LSTM_LAYERS):
        dl = tf.keras.layers.LSTM(
            N_LSTM_UNITS,
            return_sequences=True,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT_RATE,
            name=f'decoder_lstm_{i+1}'
        )
        if i == 0:
            y = dl(y, initial_state=states)
        else:
            y = dl(y)
    # (어텐션 + 후처리)
    a      = tf.keras.layers.AdditiveAttention(name='attention')([y, x])
    concat = tf.keras.layers.Concatenate()([y, a])
    td1    = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(N_LSTM_UNITS, activation='tanh')
    )(concat)
    ln     = tf.keras.layers.LayerNormalization()(td1)
    dp     = tf.keras.layers.Dropout(DROPOUT_RATE)(ln)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1), name='output'
    )(dp)

    return tf.keras.Model(encoder_inputs, outputs)

########################################################################
# 2) 모델 빌드 + 가중치 로드
########################################################################
model = build_seq2seq_flexible(
    N_LSTM_UNITS=128,
    N_LSTM_LAYERS=2,
    DROPOUT_RATE=0.2,
    RECURRENT_DROPOUT_RATE=0.2
)
model.load_weights("final_model1.h5")   # 필요 시 절대 경로로 변경

########################################################################
# 3) 이미지 저장 준비
########################################################################
save_fname = "seq2seq_dynamic.png"
save_dir   = r"C:\Users\parkw\Downloads"
os.makedirs(save_dir, exist_ok=True)
save_path  = os.path.join(save_dir, save_fname)

########################################################################
# 4) plot_model 시도 → 실패 시 대체 저장
########################################################################
try:
    plot_model(
        model,
        to_file=save_path,
        show_shapes=False,
        show_dtype=False,
        expand_nested=False,
        dpi=120,
        rankdir='LR'
    )
except Exception as e:
    print(f"plot_model 실패: {e}")

if os.path.isfile(save_path):
    print(f"✅ 이미지 저장 완료: {save_path}")
else:
    print("⚠️ plot_model로 저장되지 않아, 대체 방법 실행합니다.")
    dot = model_to_dot(
        model,
        show_shapes=False,
        show_dtype=False,
        expand_nested=False,
        rankdir='LR'
    )
    dot.write_png(save_path)
    print(f"✅ 대체 방법으로 이미지 저장 완료: {save_path}")
