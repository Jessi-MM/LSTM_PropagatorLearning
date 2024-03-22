# LSTM to solve the Time Dependent Schrödinger Equation

We implemented an LSTM model in PyTorch to propagate wave packets under certain kinds of time-dependent potentials through time.

![Density&Potential](./src/Animation/gifs/animation-dens&pot.gif)

## Table of Contents
1. [Data Generation](#datagen)
2. [Architecture of the LSTM model](#arch)
3. [Results and Predictions](#results)
4. [Previous Works](#prevw)
5. [About Physical System](#phys)

<a name="datagen"></a>
## Data Generation 
Our data is in the file `Data_Gaussian.tar.xz`.
LSTM model uses wave packet and potential at time $t$ as input and returns the wave packet at time $t+ \Delta t$ under potential given. We used an LSTM model because we aborded this problem as a time series problem.
![In-Ou](img/dataInputOutput.png)

To generate wave packets: $\psi(r,t)$ propagated in time we applied the DVR method to solve the Time Dependent Schrödinger Equation, using the *Proton Transfer System* as model of potential. We used a grid of $n=32$ points in the position space: $[-1.5,1.5] \AA $, each step of time was: $\Delta t= 1 fs$, and generated trajectories of $200fs$ (this is the sequence of lenght to the LSTM). We used 3290 trajectories to train the model. The next figure shows an example of what contains one trajectory.

![Trajectory](img/DiagTrayectoria.png)

The notebook to generate data is in `src/Proton_Transfer_DataGenerate.ipynb` and contains a guide to use the class.

<a name="arch"></a>
## Architecture
The proccesing of data, details of the LSTM model, and all the steps of the training and testing are in the notebook:  `src/ANN_models/LSTM-Model2.ipynb`.

Sumary of model:

| Layers | Features | Input/Output |
|--------|----------|--------------|
| LSTM   | 1024     | (96,1024)    |
| LSTM   | 1024     | (1024,1024)  |
| Linear | 64       | (1024,64)    |

- Loss function: MSE
- Optimizer: AdamW, weight_decay=0.01
- learning rate: 1e-4
- batch size: 10
- epochs: 90

<a name="results"></a>
## Results and Predictions
The trained model was saved in: `src/ANN_Models/4700Data_LSTM_MODEL2/14-05-23_10EPOCHS.pth`

We obtained an accuracy magnitud of 88% over a test set with 470 trajectories. The next figures shows some predictions by LSTM model.

![Dens&Potgif](./src/Animacion/animationLSTM-dens&pot.gif)
![Predictions_step](img/1step.png)
![Predictions_Traj](img/trajDens.png)

<a name="prevw"></a>
## Previous Works
 [Artificial Neural Networks as Propagators in Quantum Dynamics](https://doi.org/10.1021/acs.jpclett.1c03117)
 
<a name="phys"></a>
## About Physical System
:snail:


