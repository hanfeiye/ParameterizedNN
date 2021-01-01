#!/usr/bin/env python
# ******************************************************
# Author       : Hanfei Ye
# Last modified: 2020-12-31 14:48
# Email        : hanfei.ye@cern.ch
# Filename     : model.py
# Description  : 
# ******************************************************

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def build_model(attribute_dim):
  # attribute_dim: dimention of samples' attributes
  model = Sequential()
  model.add(Dense(64, input_dim=attribute_dim, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model
