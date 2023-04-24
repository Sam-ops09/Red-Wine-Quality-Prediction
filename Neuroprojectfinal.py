import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

# Load the data
data = pd.read_csv('winequality-red.csv', delimiter=';')

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(data.drop('quality', axis=1), data['quality'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping])

# Evaluate the model on the testing set
loss = model.evaluate(X_test_scaled, y_test)
print('Mean squared error on testing set:', loss)


# Predict wine quality for the testing set
y_pred = model.predict(X_test_scaled)

# Calculate the percentage of good quality wines in the testing set
threshold = 6
num_good = sum(y_test >= threshold)
num_total = len(y_test)
num_pred_good = sum(y_pred.reshape(-1) >= threshold)
percent_good_actual = num_good / num_total * 100
percent_good_pred = num_pred_good / num_total * 100

print('Actual percentage of good quality wines:', percent_good_actual)
print('Predicted percentage of good quality wines:', percent_good_pred)
