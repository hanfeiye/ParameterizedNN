# ParameterizedNN
Simple implementation of parameterized neural network (PNN) by using Keras.

It runs on 114.212.235.193 since Bowen Zhang ( one of members of NJU HEP group ) has installed Keras and its depencence on it(Thanks, Bowen!). 

Notes:
+ If you are interested in Keras, please refer to this page. [Keras](https://keras.io/zh/)
+ Information on what is artificial neural network and how it works can be found on Internet. 


## Log and Clone this repo.
First, log on 114.212.235.193 by your own account, UserName@114.212.235.193

In NJU, use

```bash
ssh -Y UserName@114.212.235.193
```

Outside of NJU, use

```bash
ssh -p 6000 UserName@42.193.179.181
```

Then, create an empty folder for your work.

```bash
mkdir YourWorkFolder/
```

And Clone the repo.

```bash
git clone https://github.com/hanfeiye/ParameterizedNN.git
cd ParameterizedNN
git checkout -b myDevBranch
```

## Set up the environment

```bash
source setup.sh
```

## Prepare your inputs
Prepare your inputs by yourself. They are ROOT files sotring trees(TTree).

+ Signal truth mass should be assigned to each background or signal events. 
+ All of background or signal events should be labeled (0 for bkg, 1 for sig).

Refer to this [script](doc/copytree_addSignalMass.py) to see how these two points are implemented.

A few inputs has been provided and can be found in inputs/

```bash
ls -l inputs/
```

## Train and test

```bash
python train_PNN.py
```

Definition of NN model and settings for plotting can be found in source/ folder.

```bash
ls -l source/
```

You can modify them according to your needs.

## Hyper-paramerters tuning

+ the number of epochs  
+ batch size
+ the number of units in model
+ ...

...on going.


## Application

...on going. (transfer output arrays into a new branch of a TTree, stored as ROOT files)


## Examples
A PNN is trained by using signal masses of 400, 600, 800, 1000 and 1500 GeV. It's tested at 700, 900, 1200 GeV.

ROC plot:

<img src=examples/plots/Validation_ROCs.png>

Output plot for 1500 GeV:

<img src=examples/plots/Validation_Dis_1500GeV.png>

