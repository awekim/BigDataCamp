# 필요 패키지 셋팅
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models


# 데이터 불러오기
df = pd.read_csv("coin_data.csv")


# 데이터 전처리
# 거래량 열을 보면 string으로 되어있고, K, M, B단위로 되어 있는데 이것들을 실제 숫자로 바꿔주는 함수입니다.
def convert_volume(value):
    if 'K' in value:
        return float(value.replace('K', '')) * 1000
    elif 'M' in value:
        return float(value.replace('M', '')) * 1000000
    elif 'B' in value:
        return float(value.replace('B', '')) * 1000000000
    else:
        return float(value)

# 위에서 만든 함수로 거래량 데이터를 변환합니다.
df['거래량'] = df['거래량'].apply(convert_volume)


# 날짜 삭제
# 날짜열 삭제 및 그대로 df에 적용
df.drop(['날짜'], axis=1, inplace=True)


# , % 없애고 숫자로 변환하는 함수
def preprocess_value(value):
    return float(value.replace(',', '').replace('%', ''))

# "종가", "시가", "고가", "저가", "변동 %" 열에 대하여 변환 함수들 적용
for column in ["종가", "시가", "고가", "저가", "변동 %"]:
    df[column] = df[column].apply(preprocess_value)

# 데이터 정규화
# min-max scaler 이용
scaler = MinMaxScaler()

# scaler를 이용하여 df에서 min, max값 탐색
scaler.fit(df)

# 변환한 scaler 저장
joblib.dump(scaler, 'scaler.pkl')

# 위에서 fit 시킨 scaler를 이용하여 데이터 프레임 변환 후 scaled_data에 저장
scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

X = scaled_data.drop(["변동 %"], axis=1)
y = scaled_data["변동 %"]

# 특성과 타겟 분리
print(X)
print(y)


# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 모델 생성 및 학습
model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(5,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# EarlyStopping 콜백 설정
early_stopping = EarlyStopping(monitor='loss', patience=200, min_delta=0.01)

# 모델 학습
model.fit(X_train, y_train, epochs=1000, callbacks=[early_stopping])


# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 모델 파일 이름
model_filename = 'coin_linear_regression_model.pkl'
scaler_name = 'scaler.pkl'

# 모델을 저장합니다.
print('====================Model Save====================')
print(f'Model saved as {model_filename}')
joblib.dump(model, model_filename)
print('==================================================')

loaded_model = joblib.load(model_filename)
scaler = joblib.load(scaler_name)

test_data = pd.DataFrame({
    '종가': [500],
    '시가': [400],
    '고가': [550],
    '저가': [400],
    '거래량': [40000],
    '변동 %': [0]
})

scaled_test_data = scaler.transform(test_data)
print(scaled_test_data[:,:-1])

predicted_volatality = loaded_model.predict(scaled_test_data[:,:-1])

print(predicted_volatality)

# 역변환을 위해 원래의 최솟값과 최댓값을 저장
original_min = scaler.data_min_
original_max = scaler.data_max_

# 역변환 수행
unscaled_data = predicted_volatality[0][0] * (original_max[-1] - original_min[-1]) + original_min[-1]

print(f'향후 주식 변동성 예측: {unscaled_data}%')