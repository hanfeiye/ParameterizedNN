# ParameterizedNN
Implementation of parameterized neural network (PNN) by using Keras.

It runs on 114.212.235.193 since Bowen Zhang ( one of members of NJU HEP group ) has installed Keras and its depencence on it(Thanks, Bowen!). 

+ Note: if you are interested in Keras, please refer to this page. [Keras](https://keras.io/zh/)

## Log and Clone this repo.
First, log on 114.212.235.193 by your own account, UserName@114.212.235.193

```bash
ssh -Y UserName@114.212.235.193
```
+ Note: this machine is in 526, Physics Building, NJU.


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

## Prepare your inputs
Prepare your inputs by yourself. They are ROOT files sotring trees(TTree).

+ Signal truth mass should be assigned to each background or signal events. 
+ All of background or signal events should be labeled (0 for bkg, 1 for sig).

Refer to this [script](doc/copytree_addSignalMass.py) to see how these two points are implemented.

## Train and test


## Application
...on going.

