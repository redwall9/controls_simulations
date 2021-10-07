from models.InvertedPendulumModel import InvertedPendulumModel
from controls.EKF import EKF
from controls.MPC import MPC

import sys
import scipy
import scipy.linalg
import numpy as np
import math
import matplotlib.pyplot as plt

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
# variance of the random acceleration is 0.1 m/s^2
V = np.array([
    [Ts**4 / 3, Ts**3 / 2, 0, 0],
    [Ts**3 / 2, Ts**2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]) * 0.1

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
Q[1][1] = 1
Q[2][2] = 10
Q[3][3] = 1

R = np.identity(1)

# initial state covariance estimate for EKF
P0 = np.identity(4) * 0.1

model = InvertedPendulumModel(0.5, 0.2, 0.1, 0.3, 0.006, x, u, V, W, Ts)
ekf = EKF(model, x, P0)

sim_length = 1000 + L                   # in number of time steps

# matrices to store the true state, estimated state, and reference trajectory
x_true = np.empty((4, sim_length - L))
x_hat = np.empty((4, sim_length - L))
x_ref = np.empty((4, sim_length - L))

# generate the trajectory that we want to follow
for i in range(0, sim_length - L):
    x_ref[0, i] = .125 * math.sin(math.pi * Ts * i / 5)
    x_ref[1, i] = .125 * math.pi / 5 * math.cos(math.pi * Ts * i / 5)
    x_ref[2, i] = 0
    x_ref[3, i] = 0

for i in range(sim_length - L):
    x_true[:, i] = x[:,0]
    x_hat[:, i] = ekf.x_hat[:,0]

    # update true model for system based on the currently applied input
    x = x + model.get_noisy_model_derivative(x, u)*Ts

    # update EKF estimate using currently applied input and (noisy) output
    ekf.predict_and_update(u, model.get_noisy_output(x))

    # compute new control input using model predictive control
    u = np.array([[MPC(model, ekf.x_hat, x_ref[:,i:i+L].flatten("F").reshape(-1, 1), Q, R)[0]]])

# plot the results
time = np.arange(0, Ts*(sim_length-L), Ts)

plt.plot(time, x_true[0, :])
plt.plot(time, x_ref[0, :])
plt.plot(time, x_hat[0, :])
plt.show()