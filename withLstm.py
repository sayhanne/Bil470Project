from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

training_data = pd.read_csv("train_x.csv")
training_targets = pd.read_csv("train_y_class.csv")
test_data = pd.read_csv("test_x.csv")
test_targets = pd.read_csv("test_y_class.csv")

train_np = training_data.to_numpy()
test_np = test_data.to_numpy()
train_target_np = training_targets.to_numpy()
test_target_np = test_targets.to_numpy()

####normalization
scaler = MinMaxScaler(feature_range=(0, 1))
train_np = scaler.fit_transform(train_np)
test_np = scaler.fit_transform(test_np)

train_np = train_np.reshape((train_np.shape[0], train_np.shape[1], 1))
test_np = test_np.reshape((test_np.shape[0], test_np.shape[1], 1))

#Model
lstm_model = Sequential()

lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (train_np.shape[1], train_np.shape[2])))
lstm_model.add(LSTM(units = 50, return_sequences = False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(units = 1, kernel_initializer='uniform', activation='linear'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy'])

lstm_model.fit(train_np, train_target_np, batch_size = 28, epochs = 100)

score, acc = lstm_model.evaluate(test_np, test_target_np,
                            batch_size=14)
print('Test score:', score)
print('Test accuracy:', acc)

test_estimation=lstm_model.predict(test_np)

test_np = test_np.reshape((test_np.shape[0], test_np.shape[1]))
inv_test_estimation = scaler.inverse_transform(test_np)
inv_test_estimation = inv_test_estimation[:,0]

inv_test_target = test_target_np


#test
total = 0;
y_cap_arr = []
for i in range(len(inv_test_estimation)):
  y_cap_arr.append(1 if inv_test_estimation[i]>0 else 0)
  # print(y_cap_arr[i], inv_test_target[i])
  if  y_cap_arr[i] == inv_test_target[i]:
    total+=1

print(total/len(y_cap_arr))

###draw
fig = figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
y_cap = plt.plot(y_cap_arr, 'g--', label="estimation")
y = plt.plot(test_targets, 'k', label="targets")

scale_factor = 5
plt.legend(loc="upper left")
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * 1, xmax * 1)
plt.ylim(ymin * 30, ymax * 2.5)

plt.show()