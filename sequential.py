import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


training_data = pd.read_csv("train_x.csv")
training_targets = pd.read_csv("train_y_class.csv")
test_data = pd.read_csv("test_x.csv")
test_targets = pd.read_csv("test_y_class.csv")

classifier = Sequential([
  Dense(units=20, activation='relu', input_dim=10),
  Dense(units=30, activation='sigmoid'),
  Dense(units=10, activation='sigmoid'),
  Dense(units=5, activation='tanh'),
  Dense(1, kernel_initializer='uniform', activation='linear')                         
])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(training_data, training_targets, batch_size = 5, epochs = 100)

score, acc = classifier.evaluate(test_data, test_targets,
                            batch_size=5)
print('Test score:', score)
print('Test accuracy:', acc)


test_pred = classifier.predict(test_data,batch_size=5)
total = 0
test_pred_class = []
for i in range(len(test_pred)):
  if test_pred[i] > 0.41:
    test_pred_class.append(1)
  else:
    test_pred_class.append(0)
  # print(test_pred_class[i], int(test_targets.iloc[i]))
  if  test_pred_class[i] == int(test_targets.iloc[i]):
    total+=1

print(total/len(test_pred_class))



fig = figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
y_cap = plt.plot(test_pred_class, label="estimation")
y = plt.plot(test_targets, 'rx', label="targets")
plt.legend(loc="upper left")
scale_factor = 5

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * 1, xmax * 1)
plt.ylim(ymin * 30, ymax * 2.5)

plt.show()