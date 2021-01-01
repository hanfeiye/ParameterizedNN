#!/usr/bin/env python
# ******************************************************
# Author       : Hanfei Ye
# Last modified: 2020-12-31 15:17
# Email        : hanfei.ye@cern.ch
# Filename     : train_PNN.py
# Description  : 
# ******************************************************

from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_curve
from sklearn.metrics         import auc

import uproot
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


import matplotlib.pyplot as plt

from source.model import build_model 
from source.tools import tree2array, training_and_test_set, retrieve_weight, predict_array
from source.tools import validation_set, retrieve_weight_val
from source.plot  import plot_roc, plot_output_distribution 

class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.losses = {'batch':[], 'epoch':[]}
    self.accuracy = {'batch':[], 'epoch':[]}
    self.val_loss = {'batch':[], 'epoch':[]}
    self.val_acc = {'batch':[], 'epoch':[]}
 
  def on_batch_end(self, batch, logs={}):
    self.losses['batch'].append(logs.get('loss'))
    self.accuracy['batch'].append(logs.get('accuracy'))
    self.val_loss['batch'].append(logs.get('val_loss'))
    self.val_acc['batch'].append(logs.get('val_accuracy'))
 
  def on_epoch_end(self, batch, logs={}):
    self.losses['epoch'].append(logs.get('loss'))
    self.accuracy['epoch'].append(logs.get('accuracy'))
    self.val_loss['epoch'].append(logs.get('val_loss'))
    self.val_acc['epoch'].append(logs.get('val_accuracy'))
 
  def loss_plot(self, loss_type):
    iters = range(len(self.losses[loss_type]))
    plt.figure()
    # loss
    plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
    if loss_type == 'epoch':
      # val_loss
      plt.plot(iters, self.val_loss[loss_type], 'b', label='val loss')
    plt.grid(True)
    plt.xlabel(loss_type)
    plt.ylabel('loss')
    plt.ylim(0.,1.)
    plt.legend(loc="best")
    plt.savefig("Loss_{}_plot.png".format(loss_type))
    plt.close()

  def acc_plot(self, loss_type):
    iters = range(len(self.losses[loss_type]))
    plt.figure()
    # acc
    plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
    if loss_type == 'epoch':
      # val_acc
      plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
    plt.grid(True)
    plt.xlabel(loss_type)
    plt.ylabel('acc')
    plt.legend(loc="best")
    plt.savefig("Acc_{}_plot.png".format(loss_type))
    plt.close()


if __name__ == '__main__':

  mass_list = [700, 900, 1200, 1500]

  n_epochs = 1000
  batch_size = 256

  # set up seed
  seed = 8
  np.random.seed(seed)
  ####################
  # Train and evaluate
  ####################
  infile = uproot.open("./inputs/TrainingPNNInputs_c16d.root")
  print(">>>Show the keys of input files: {}".format(infile.keys()))

  X, Y, attribute_dim = tree2array(infile)

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
  # weight arrays
  weight_train, weight_test = retrieve_weight(X_train, X_test, attribute_dim)
  # taining and test sets
  x_train, x_test = training_and_test_set(X_train, X_test, attribute_dim)

  history = LossHistory()
  # create model
  model = build_model(attribute_dim)
  # fit model
  model.fit(x_train, y_train, sample_weight=weight_train, validation_data=(x_test, y_test), epochs=n_epochs, batch_size=batch_size, callbacks=[history])
  # evaluate model
  score = model.evaluate(x_test, y_test, sample_weight=weight_test, batch_size=128)
  print(">>>Evaluation Score (accuracy): {}".format(score[1]))

  history.acc_plot('epoch')
  history.loss_plot('epoch')
  history.acc_plot('batch')
  history.loss_plot('batch')
  ###################
  # Validation
  ###################
  for mass in mass_list:
    test_file = uproot.open("./inputs/Validation_{}GeV_c16d.root".format(mass))

    X_val, y_val, attribute_dim_val = tree2array(test_file)
    weight_val = retrieve_weight_val(X_val, attribute_dim_val)
    x_val = validation_set(X_val, attribute_dim_val)

    # predict
    y_pred = model.predict(x_val).ravel()
    bkg_pred_array, sig_pred_array, bkg_weights_array, sig_weights_array = predict_array(y_pred, y_val, weight_val)
    # get fake and true positive rate
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    # calculate AUC
    auc_value = auc(fpr,tpr)
    # plot ROC
    plt.figure(0)
    plot_roc(plt, fpr, tpr, auc_value, mass)
    # plot output distribution
    plt.figure(1)
    plot_output_distribution(plt, bkg_pred_array, sig_pred_array, bkg_weights_array, sig_weights_array, mass)
    plt.close('all')

  # plot ROCs together
  plt.figure(0)
  plt.plot([0,1], [0,1], 'k--') # black dash line
  for mass in mass_list:
    test_file = uproot.open("./inputs/Validation_{}GeV_c16d.root".format(mass))

    X_val, y_val, attribute_dim_val = tree2array(test_file)
    weight_val = retrieve_weight_val(X_val, attribute_dim_val)
    x_val = validation_set(X_val, attribute_dim_val)

    # predict
    y_pred = model.predict(x_val).ravel()
    bkg_pred_array, sig_pred_array, bkg_weights_array, sig_weights_array = predict_array(y_pred, y_val, weight_val)
    # get fake and true positive rate
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    # calculate AUC
    auc_value = auc(fpr,tpr)
    plt.plot(fpr, tpr, label="$m_X$ = {:s} GeV ({:.3f})".format(str(mass),auc_value))
  plt.xlabel("Fake Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC curve: PNN")
  plt.legend(loc="best")
  plt.savefig("Validation_ROCs.png")
  plt.close('all')
