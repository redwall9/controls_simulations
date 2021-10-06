from models.InvertedPendulumModel import InvertedPendulumModel
from controls.EKF import EKF
from controls.MPC import MPC

import sys
import scipy
import scipy.linalg
import numpy as np
import math

# make numpy output a little easier to read
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=3)

# iniitial state and input vectors
x = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.0]
])

u = np.array([[0]])

# sampling rate
Ts = 0.01

# length of the prediction horizon for model preditive control
L = 15

# state covariance matrices. We use the continuous time random acceleration
# model as defined in
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7289336, where the
# variance of the random acceleration is 0.001 m/s^2
V = np.array([
    [Ts**4 / 3, Ts**3 / 2, 0, 0],
    [Ts**3 / 2, Ts**2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]) * 0.001

# output / sensor covariance matrix. The first entry on the diagonal represents
# the variance of the position sensor for the cart. A more intuituve way to look
# at it is in terms of the standard deviation, which is 0.01 or 10 cm. Any
# decent encoder should easily be able to be measure to within 10 cm, so this is
# a pretty consdervative assumption. Similarly, the second diagonal entry is the
# variance of the encoder measuring the pendulum angle. A standard deviation of
# 0.01 radians is 0.57 degrees, which again is reasonable for an encoder.
W = np.array([
    [0.0001, 0],
    [0, 0.0001]
])

# quadratic cost function weighting matrices. Q is the weight on the state, and
# we weight the position (Q[0][0]) and deviation angle (Q[2][2]) heavier since
# those are the state variables we really care about maintaining. R is the
# weight on the control effort (input)
Q = np.zeros((4,4))
Q[0][0] = 10
Q[1][1] = 5
Q[2][2] = 10
Q[3][3] = 1

R = np.identity(1)

# initial state covariance estimate for EKF
P0 = np.identity(4) * 0.1

model = InvertedPendulumModel(0.5, 0.2, 0.1, 0.3, 0.006, x, u, V, W, Ts)
ekf = EKF(model, x, P0)

# generate the trajectory that we want to follow - a simple line where the cart
# is moving at 0.05 m/s
sim_length = 1000                   # in number of time steps
x_ref = np.zeros((sim_length*4,1))
for i in range(0, sim_length*4, 4):
    x_ref[i][0]   = 0.05 * (Ts * i / 4)
    x_ref[i+1][0] = 0.05
    x_ref[i+2][0] = 0
    x_ref[i+3][0] = 0

for i in range(sim_length - L):
    # update true model for system based on the currently applied input
    x = x + model.get_noisy_model_derivative(x, u)*Ts

    # update EKF estimate using currently applied input and (noisy) output
    ekf.predict_and_update(u, model.get_noisy_output(x))

    # compute new control input using model predictive control
    u = np.array([[MPC(model, ekf.x_hat, x_ref[i*4:(i+L)*4,:], Q, R)[0]]])

    print(x)
    print(x_ref[i*4])