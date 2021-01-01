#!/usr/bin/env python
# ******************************************************
# Author       : Hanfei Ye
# Last modified: 2020-12-30 17:03
# Email        : hanfei.ye@cern.ch
# Filename     : PNN.py
# Description  : 
# ******************************************************
import uproot
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Data preparation, (X, Y)
infile = uproot.open("./inputs/TrainingInputs_c16d.root")
print(">>>Show the keys of input files: {}".format(infile.keys()))

tree = infile['NOMINAL']
Y = tree.array('signal_label')
print(">>>Shape of labels array: {}".format(Y.shape))

branch_list = [
               'SumOfPt',
               'lephad_dphi',
               'lephad_met_lep0_cos_dphi',
               'lephad_met_lep1_cos_dphi',
               'lep_pt',
               'tau_pt',
               'bjet_pt',
               'MET',
               'signal_mass',
               'weight',
              ]

dics = tree.arrays(branch_list, namedecode="utf-8")
print(">>>Show the dictionary of attributes: ")
print(dics)

attribute_dim = int(len(dics))-1
print(">>>Dimention of attribute space: {}".format(attribute_dim))

tmp_list = [] 
for key, value in dics.items():
  tmp_list.append(value)
print(">>>Show the list of attributes: ")
print(tmp_list)
X = np.array(tmp_list)
X = X.T # reverse the matrix
print(">>>Shape of attributes array: {}".format(X.shape))


# set up seed
seed = 8
np.random.seed(seed)

# split train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
weight_train = X_train[:,len(dics)-1]
weight_test = X_test[:,len(dics)-1]
x_train = X_train[:,0:len(dics)-1]
print(x_train.shape)
x_test = X_test[:,0:len(dics)-1]

# create model
model = Sequential()
model.add(Dense(64, input_dim=attribute_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(x_train, y_train, sample_weight=weight_train, epochs=20, batch_size=128)

# evaluate
score = model.evaluate(x_test, y_test, sample_weight=weight_test, batch_size=128)
print(">>>Results of evaluation: {}".format(score))

# predict

y_pred = model.predict(x_test).ravel()
print(y_pred.shape)
print(y_test.shape)
num_c = y_pred.shape[0]
print(num_c)
bkg_pred = []
sig_pred = []
sig_weights = []
bkg_weights = []
for i in range(0,num_c):
  if 1 == y_test[i]:
    sig_pred.append(y_pred[i])
    sig_weights.append(weight_test[i])
  elif 0 == y_test[i]:
    bkg_pred.append(y_pred[i])
    bkg_weights.append(weight_test[i])
print(len(bkg_pred))
print(len(sig_pred))
bkg_pred_array = np.array(bkg_pred)  
sig_pred_array = np.array(sig_pred)  
bkg_weights_array = np.array(bkg_weights)
sig_weights_array = np.array(sig_weights)

# get fake and true positive rate
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# calculate AUC
from sklearn.metrics import auc
auc_value = auc(fpr,tpr)

# plot ROC
import matplotlib.pyplot as plt
plt.figure(0)
plt.plot([0,1], [0,1], 'k--') # black dash line
plt.plot(fpr, tpr, label="Binary NN (area = {:.3f})".format(auc_value))
plt.xlabel("Fake Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve: NN")
plt.legend(loc="best")

# plot output distribution
plt.figure(1)
plt.hist(bkg_pred_array, bins=10, range=(0,1), 
         density=1, weights=bkg_weights_array, facecolor="blue", 
         edgecolor="black", alpha=0.7,
         label="bkg")
plt.hist(sig_pred_array, bins=10, range=(0,1), 
         density=1, weights=sig_weights_array, facecolor="red", 
         edgecolor="black", alpha=0.7,
         label="sig")
plt.xlabel("NN outputs")
plt.ylabel("Entries")
plt.legend(loc="best")
plt.show()



