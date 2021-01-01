#!/usr/bin/env python
# ******************************************************
# Author       : Hanfei Ye
# Last modified: 2020-12-31 14:54
# Email        : hanfei.ye@cern.ch
# Filename     : plot.py
# Description  : 
# ******************************************************

import matplotlib.pyplot as plt

def plot_roc(plt, fpr, tpr, auc_value, mass):
  plt.plot([0,1], [0,1], 'k--') # black dash line
  plt.plot(fpr, tpr, label="$m_X$ = {:s} GeV (auc = {:.3f})".format(str(mass),auc_value))
  plt.xlabel("Fake Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC curve: PNN")
  plt.legend(loc="best")
  plt.savefig("Validation_ROC_{}GeV.png".format(mass))

def plot_output_distribution(plt, bkg_pred_array, sig_pred_array, bkg_weights_array, sig_weights_array, mass):
  plt.hist(bkg_pred_array, bins=10, range=(0,1),
           density=1, weights=bkg_weights_array, facecolor="blue",
           edgecolor="black", alpha=0.7,
           label="ttbar")
  plt.hist(sig_pred_array, bins=10, range=(0,1),
           density=1, weights=sig_weights_array, facecolor="red",
           edgecolor="black", alpha=0.7,
           label="$m_X$ = {:s} GeV".format(str(mass)))
  plt.xlabel("PNN outputs")
  plt.ylabel("Entries")
  plt.legend(loc="best")
  plt.savefig("Validation_Dis_{}GeV.png".format(mass))


