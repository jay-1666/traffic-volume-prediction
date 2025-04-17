import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv(r"C:\Users\jay\Downloads\archive (17)\Banglore_traffic_Dataset.csv")

df['Date'] = pd.to_datetime(df['Date'])

cities = df['Area Name'].unique()

window_size = 7

predictions = []

for city in cities:
    city_df = df[df['Area Name'] == city].sort_values('Date')

    city_df = city_df.drop(['Area Name', 'Road/Intersection Name'], axis=1)

    city_df = pd.get_dummies(city_df)

    city_df = city_df.fillna(0)

    if 'Traffic Volume' not in city_df.columns:
        continue

    target_col = 'Traffic Volume'
    feature_cols = city_df.columns.drop(['Date', target_col])

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(city_df[feature_cols])
    scaled_target = scaler.fit_transform(city_df[[target_col]])

    X = []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i-window_size:i])
    
    if len(X) == 0:
        continue

    X = np.array(X)
    y = scaled_target[window_size:]

    if len(X) < 10:
        continue

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=30, batch_size=16, callbacks=[early_stop], verbose=0)

    last_window = scaled_features[-window_size:].reshape(1, window_size, -1)
    next_scaled = model.predict(last_window)[0][0]

    dummy_input = np.zeros((1, len(feature_cols)))
    dummy_input[0, -1] = next_scaled  
    predicted_volume = scaler.inverse_transform(dummy_input)[0, -1]

    predictions.append({'City': city, 'Next Day Predicted Traffic Volume': predicted_volume})

result_df = pd.DataFrame(predictions)
print(result_df)
