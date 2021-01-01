#!/usr/bin/env python
# ******************************************************
# Author       : Hanfei Ye
# Last modified: 2020-12-31 15:02
# Email        : hanfei.ye@cern.ch
# Filename     : tools.py
# Description  : 
# ******************************************************

import uproot
import numpy as np

def tree2array(infile):
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
                 'lep_eta',
                 'tau_eta',
                 'bjet_eta',
                 'lep_phi',
                 'tau_phi',
                 'bjet_phi',
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

  return X, Y, attribute_dim

def training_and_test_set(X_train, X_test, attribute_dim):
  x_train = X_train[:,0:attribute_dim]
  print(x_train.shape)
  x_test = X_test[:,0:attribute_dim]

  return x_train, x_test

def validation_set(X_val, attribute_dim_val):
  x_val = X_val[:,0:attribute_dim_val]
  print(x_val.shape)

  return x_val


def retrieve_weight(X_train, X_test, attribute_dim):
  weight_train = X_train[:,attribute_dim]
  weight_test = X_test[:,attribute_dim]

  return weight_train, weight_test

def retrieve_weight_val(X_val, attribute_dim_val):
  weight_val = X_val[:,attribute_dim_val]

  return weight_val

def predict_array(y_pred, y_test, weight_test):
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

  return bkg_pred_array, sig_pred_array, bkg_weights_array, sig_weights_array
