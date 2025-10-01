import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Input, RepeatVector, TimeDistributed,
    Dropout, AdditiveAttention, LayerNormalization,
    Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ----------------------------
# --- 0. 하이퍼파라미터 ---
# ----------------------------
N_HOURS_INPUT = 50
N_HOURS_TOTAL = 400
N_HOURS_PREDICT = N_HOURS_TOTAL - N_HOURS_INPUT
N_FEATURES = 1
N_LSTM_UNITS = 128
N_LSTM_LAYERS = 2
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT_RATE = 0.2
N_EPOCHS = 150
N_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15

# ----------------------------
# --- 1. 가상 데이터 준비 ---
# ----------------------------
def load_banana_data(num_samples=104, total_hours=400):
    """가상의 바나나 신선도 시계열 데이터 생성."""
    all_data = []
    time = np.arange(total_hours)
    for i in range(num_samples):
        initial_decay_rate = np.random.uniform(0.001, 0.01)
        overall_decay_speed = np.random.uniform(0.005, 0.015)
        mid_point = total_hours / 2 + np.random.uniform(-50, 50)
        # 로지스틱 형태
        freshness = 1 / (1 + np.exp(overall_decay_speed * (time - mid_point)))
        # 초반부(입력 구간)만 따로 초기 drop 처리
        initial_drop = 1 - initial_decay_rate * time[:N_HOURS_INPUT]
        freshness[:N_HOURS_INPUT] = np.minimum(freshness[:N_HOURS_INPUT], initial_drop)
        # 노이즈 추가 및 범위 클리핑
        freshness = np.clip(freshness + np.random.normal(0, 0.02, total_hours), 0, 1)
        all_data.append(freshness)
    return np.array(all_data)

all_freshness_data = load_banana_data(num_samples=104, total_hours=N_HOURS_TOTAL)
X = all_freshness_data[:, :N_HOURS_INPUT].reshape((-1, N_HOURS_INPUT, N_FEATURES))
Y = all_freshness_data[:, N_HOURS_INPUT:].reshape((-1, N_HOURS_PREDICT, N_FEATURES))

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"학습 데이터 X: {X_train.shape}, Y: {Y_train.shape}")
print(f"검증 데이터 X: {X_val.shape}, Y: {Y_val.shape}")

# ----------------------------
# --- 2. 개선된 Seq2Seq + Attention ---
#     (Decoder LSTM 출력 & Attention 결합)
# ----------------------------

# [인코더]
encoder_inputs = Input(shape=(N_HOURS_INPUT, N_FEATURES), name='encoder_input')
encoder_lstm_outputs = encoder_inputs
encoder_states_list = []

for i in range(N_LSTM_LAYERS):
    is_last_layer = (i == N_LSTM_LAYERS - 1)
    encoder_lstm = LSTM(N_LSTM_UNITS,
                        return_sequences=True,
                        return_state=True,
                        dropout=DROPOUT_RATE,
                        recurrent_dropout=RECURRENT_DROPOUT_RATE,
                        name=f'encoder_lstm_{i+1}')
    encoder_lstm_outputs, state_h, state_c = encoder_lstm(encoder_lstm_outputs)
    if is_last_layer:
        encoder_states_list = [state_h, state_c]

encoder_outputs_seq = encoder_lstm_outputs  # 인코더 모든 타임스텝 출력 (Attention용)

# [디코더]
# 실제 디코더 입력(TF) 미사용 -> placeholder
decoder_inputs = Input(shape=(N_HOURS_PREDICT, N_FEATURES), name='decoder_input_placeholder')

# 인코더 마지막 상태 -> 디코더 첫 RepeatVector로
decoder_lstm_outputs = RepeatVector(N_HOURS_PREDICT)(encoder_states_list[0])
decoder_states = encoder_states_list

for i in range(N_LSTM_LAYERS):
    decoder_lstm = LSTM(N_LSTM_UNITS,
                        return_sequences=True,
                        return_state=False,
                        dropout=DROPOUT_RATE,
                        recurrent_dropout=RECURRENT_DROPOUT_RATE,
                        name=f'decoder_lstm_{i+1}')
    # 첫 디코더층만 인코더 상태를 초기 상태로 사용
    init_state = decoder_states if i == 0 else None
    decoder_lstm_outputs = decoder_lstm(decoder_lstm_outputs, initial_state=init_state)

# [Attention 레이어]
attention_layer = AdditiveAttention(name='attention_layer')
attention_result = attention_layer([decoder_lstm_outputs, encoder_outputs_seq])
# shape: (batch, N_HOURS_PREDICT, N_LSTM_UNITS)

# =============================
# 2번 개선점: "디코더 LSTM 출력" + "Attention 결과" 결합
# =============================
concat_output = Concatenate(axis=-1)([decoder_lstm_outputs, attention_result])
# concat_output shape: (batch, N_HOURS_PREDICT, 2 * N_LSTM_UNITS)

decoder_combined_dense = TimeDistributed(
    Dense(N_LSTM_UNITS, activation='tanh')
)(concat_output)
# shape: (batch, N_HOURS_PREDICT, N_LSTM_UNITS)

# Layer Normalization
norm_layer = LayerNormalization()
normalized_output = norm_layer(decoder_combined_dense)

# Dropout
dropout_layer = Dropout(DROPOUT_RATE)
dropout_output = dropout_layer(normalized_output)

# 최종 출력 Dense
output_dense = TimeDistributed(Dense(N_FEATURES), name='output_dense')
decoder_outputs = output_dense(dropout_output)

# 최종 모델 (디코더 입력은 사용 안 하므로 encoder_inputs만 연결)
model = Model(encoder_inputs, decoder_outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

# ----------------------------
# --- 3. 학습 (Early Stopping) ---
# ----------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(X_train, Y_train,
                    epochs=N_EPOCHS,
                    batch_size=N_BATCH_SIZE,
                    validation_data=(X_val, Y_val),
                    callbacks=[early_stopping],
                    verbose=1)

# ----------------------------
# --- 4. 예측 함수 ---
# ----------------------------
def predict_freshness(model, initial_n_hours_data):
    """
    initial_n_hours_data: shape=(N_HOURS_INPUT,) or (N_HOURS_INPUT, 1)
    """
    if initial_n_hours_data.shape[0] != N_HOURS_INPUT:
        raise ValueError(f"입력 길이는 {N_HOURS_INPUT}이어야 합니다.")
    input_data = initial_n_hours_data.reshape((1, N_HOURS_INPUT, N_FEATURES))
    prediction = model.predict(input_data)
    return prediction.flatten()

# ----------------------------
# --- 5. 예측 예시 ---
# ----------------------------
sample_input = X_val[0].flatten()  # 검증 데이터 첫 샘플의 처음 50시간
predicted_future = predict_freshness(model, sample_input)

print(f"\n[예측 예시]")
print(f"초기 입력 (처음 10개): {sample_input[:10]}")
print(f"예측된 미래 신선도 (처음 10개): {predicted_future[:10]}")
print(f"예측 결과 길이: {predicted_future.shape}")
