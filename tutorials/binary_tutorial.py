#!/usr/bin/env python

# binary classification

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# produce virtual data
import numpy as np

n_train = 10000
n_test  = 1000

x_train = np.random.random((int(n_train),20)) # marix shape (1000,20)
y_train = np.random.randint(2, size=(int(n_train), 1)) # binary
x_test = np.random.random((int(n_test), 20)) # marix shape (100,20)
y_test = np.random.randint(2, size=(int(n_test), 1)) # binary

# build network
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20)) # 64 hidden neurons
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) 

# compile network
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# fit
model.fit(x_train, y_train,
          epochs=20, 
          batch_size=128)

# evaluate
score = model.evaluate(x_test, y_test, batch_size=128)

# predict
y_pred = model.predict(x_test).ravel()

# generate ROC data
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# calculate AUC
from sklearn.metrics import auc
auc_value = auc(fpr, tpr)

# plot ROC curve
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot([0,1], [0,1], "k--")
plt.plot(fpr, tpr, label="Binary (area = {:.3f})".format(auc_value))
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.show()
