import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, AdditiveAttention, LayerNormalization,
    Concatenate, TimeDistributed, Layer
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

########################################################################
# (A) 커스텀 Repeat 레이어: 인코더 입력 길이에 따라 (400 - n)만큼 반복
########################################################################
TOTAL_HOURS = 400

class RepeatVectorDynamic(Layer):
    """
    인코더 마지막 은닉상태 h를 (400 - 입력길이)만큼 반복하여
    디코더 입력 시퀀스를 생성하는 커스텀 레이어.
    """
    def call(self, inputs, **kwargs):
        """
        inputs = [final_h, encoder_in]
          - final_h.shape  = (batch, units)
          - encoder_in.shape = (batch, in_seq_len, features)
        반환: repeated.shape = (batch, out_seq_len, units)
          여기서 out_seq_len = (400 - in_seq_len)
        """
        final_h, encoder_in = inputs
        batch_size = tf.shape(encoder_in)[0]
        in_seq_len = tf.shape(encoder_in)[1]  # 동적 길이
        pred_len = TOTAL_HOURS - in_seq_len   # 400 - n

        # (batch, units) -> (batch, 1, units)
        final_h_expanded = tf.expand_dims(final_h, axis=1)
        # pred_len번 반복
        repeated = tf.tile(final_h_expanded, [1, pred_len, 1])
        return repeated

    def compute_output_shape(self, input_shape):
        # input_shape = [(batch, units), (batch, None, features)]
        # out shape = (batch, None, units) → None = 400 - in_seq_len (동적)
        return (input_shape[0][0], None, input_shape[0][-1])

########################################################################
# (B) 모델 구성: Encoder + Decoder(+Attention+LayerNorm+Dropout)
########################################################################
def build_seq2seq_flexible(
    N_LSTM_UNITS=128,
    N_LSTM_LAYERS=2,
    DROPOUT_RATE=0.2,
    RECURRENT_DROPOUT_RATE=0.2
):
    # -----------------------
    # 1) 인코더
    # -----------------------
    encoder_inputs = Input(shape=(None, 1), name='encoder_input')  
    # (batch, in_seq_len, 1), in_seq_len은 동적(None)

    encoder_lstm_outputs = encoder_inputs  # 다단 LSTM 쌓을 예정
    encoder_states_list = []

    for i in range(N_LSTM_LAYERS):
        is_last = (i == N_LSTM_LAYERS - 1)
        lstm_layer = LSTM(
            N_LSTM_UNITS,
            return_sequences=True,  
            return_state=True,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT_RATE,
            name=f'encoder_lstm_{i+1}'
        )
        encoder_lstm_outputs, h, c = lstm_layer(encoder_lstm_outputs)
        if is_last:
            # 마지막 LSTM 레이어의 상태만 디코더로 전달
            encoder_states_list = [h, c]

    # encoder_lstm_outputs: (batch, in_seq_len, N_LSTM_UNITS)
    # encoder_states_list: [h, c], shape=(batch, N_LSTM_UNITS)

    # -----------------------
    # 2) 디코더
    # -----------------------
    # (실제 Teacher Forcing 미사용이므로, placeholder만 만들어놓음)
    decoder_inputs = Input(shape=(None, 1), name='decoder_input_placeholder')

    # 2-1) 인코더 마지막 상태 h를 (400 - in_seq_len)만큼 반복 (커스텀 레이어)
    repeat_layer = RepeatVectorDynamic(name='dynamic_repeat')
    repeated_vector = repeat_layer([encoder_states_list[0], encoder_inputs])
    # shape=(batch, 400 - in_seq_len, N_LSTM_UNITS)

    # 2-2) 디코더 LSTM 쌓기
    decoder_lstm_outputs = repeated_vector
    for i in range(N_LSTM_LAYERS):
        dec_lstm_layer = LSTM(
            N_LSTM_UNITS,
            return_sequences=True,
            return_state=False,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT_RATE,
            name=f'decoder_lstm_{i+1}'
        )
        # 첫 디코더 레이어만 인코더 states로 초기화
        if i == 0:
            decoder_lstm_outputs = dec_lstm_layer(
                decoder_lstm_outputs,
                initial_state=encoder_states_list
            )
        else:
            decoder_lstm_outputs = dec_lstm_layer(decoder_lstm_outputs)

    # decoder_lstm_outputs: (batch, 400 - in_seq_len, N_LSTM_UNITS)

    # -----------------------
    # 3) Attention
    # -----------------------
    attention_layer = AdditiveAttention(name='attention_layer')
    attention_result = attention_layer(
        [decoder_lstm_outputs, encoder_lstm_outputs]
        # Query=decoder, Value/Key=encoder
    )
    # shape=(batch, 400 - in_seq_len, N_LSTM_UNITS)

    # 3-1) 디코더 출력 + Attention 결합
    concat_output = Concatenate(axis=-1)([decoder_lstm_outputs, attention_result])
    # shape = (batch, 400 - in_seq_len, 2*N_LSTM_UNITS)

    # 중간 Dense
    decoder_combined = TimeDistributed(Dense(N_LSTM_UNITS, activation='tanh'))(concat_output)
    
    # Layer Normalization
    norm_layer = LayerNormalization()
    normalized_output = norm_layer(decoder_combined)

    # Dropout
    dropout_layer = Dropout(DROPOUT_RATE)
    dropout_output = dropout_layer(normalized_output)

    # 최종 출력 Dense (각 타임스텝마다 1차원 = 신선도)
    output_dense = TimeDistributed(Dense(1), name='output_dense')
    decoder_outputs = output_dense(dropout_output)

    # -----------------------
    # 4) 모델 생성
    # -----------------------
    # decoder_inputs는 연결 안 함(Teacher Forcing X)
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

########################################################################
# (C) 실제 학습/사용 예시
########################################################################

# 예) 모델 빌드
model = build_seq2seq_flexible(
    N_LSTM_UNITS=128,
    N_LSTM_LAYERS=2,
    DROPOUT_RATE=0.2,
    RECURRENT_DROPOUT_RATE=0.2
)
model.summary()

# 0.952 0.948 0.943 0.943 0.942 0.940 0.938 0.935 0.931 0.928 0.926 0.923 0.918 0.910 0.898 0.882 0.875 0.870 0.866 0.860 0.857 0.853 0.850 0.839 0.832 0.826 0.820 0.811 0.799 0.781