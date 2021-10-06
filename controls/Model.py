# This abstract class is used to describe a model for a dynamic system as is
# typically used in control theory. Specifically, such a model is given as
#
#       x'(t) = f(x(t),u(t)) + v(t), y = h(x(t)) + w(t)
#
# where x is the state vector, x' is the state derivative, u is the input vector
# y is the output vector, and v and w are zero-mean Gaussian random variables.
# For this representation of a Model, the output is assumed to be independent of
# the input, which is frequently true for many practical models. In general, f
# and h are nonlinear functions. However, many control algorithms operate on a
# version of the system that is linearized about a set point. After
# linearization, the model is given by
#
#       x'(t) = Ax(t) + Bu(t), y = Cx(t)
#
# where A = df/dx, B = df/du, C = dh/dx, and all derivatives are evaluated at a
# chosen point (x0, u0). Once linearized, the model is then discretized using a
# zero-order-hold and a sample rate Ts so that the model becomes:
#
#       x[k+1] = Ad*x[k] + Bd*u[k], y[k] = C*x[k]
#
#  Classes that extend this one are required to implement
# functions that will compute the nonlinear functions f and h, as well as the
# linearized A, B, and C matrices.

import abc
import numpy as np
from numpy.random import default_rng as rng

class Model:
    # x0, u0: Nx1 and Mx1 matrices representing the state and input pair about
    #         which the model should initially be linearized
    # V:  NxN covariance matrix for v(t)
    # W:  MxM covariance matrix for w(t)
    # Ts: sampling interval for the model
    def __init__(self, x0, u0, V, W, Ts):
        # linearized model matrices
        self.Ad = None
        self.Bd = None
        self.Cd = None
        self.V = np.copy(V)
        self.W = np.copy(W)
        self.Vd = None
        self.Wd = None

        self.x0 = np.copy(x0)
        self.u0 = np.copy(u0)
        self.Ts = 0
        self.linearize_and_discretize_model(self.x0, self.u0, Ts)

    # Returns the evaluation of f(x,u)
    @abc.abstractmethod
    def get_model_derivative(self, x, u):
        pass

    def get_noisy_model_derivative(self, x, u):
        return self.get_model_derivative(x, u) +                                \
                rng().multivariate_normal(                                      \
                    np.zeros(self.V.shape[0]),                                  \
                    self.V,                                                     \
                    size=1                                                      \
                ).transpose()

    @abc.abstractmethod
    def get_output(self, x):
        pass

    def get_noisy_output(self, x):
        return self.get_output(x) +                                             \
                rng().multivariate_normal(                                      \
                    np.zeros(self.W.shape[0]),                                  \
                    self.W,                                                     \
                    size=1                                                      \
                ).transpose()

    @abc.abstractmethod
    def compute_df_dx(self, x, u):
        pass

    @abc.abstractmethod
    def compute_df_du(self, x, u):
        pass

    @abc.abstractmethod
    def compute_dh_dx(self, x, u):
        pass

    # linearizes the model about the point (x0, u0) and then discretizes it using
    # zero-order-hold with a sampling interval of Ts
    @abc.abstractmethod
    def linearize_and_discretize_model_internal(self, x0, u0, Ts):
        pass

    # use this function, and not the internal one. It does some common
    # bookkeeping and has an optimization to prevent unnecessarily
    # re-linearizing the system if the conditions haven't changed
    def linearize_and_discretize_model(self, x0, u0, Ts):
        # an optimization to avoid re-linearizing unnecessarily
        xNorm = np.linalg.norm(x0 - self.x0)
        uNorm = np.linalg.norm(u0 - self.u0)
        epsilon = 0.00000001

        if Ts == self.Ts and xNorm < epsilon and uNorm < epsilon:
            pass
        else:
            self.linearize_and_discretize_model_internal(x0, u0, Ts)
            self.x0 = x0
            self.u0 = u0
            self.Ts = Ts

            # use a first order approximation to discretize the process
            # covariance matrix, as described in:
            # https://natanaso.github.io/ece276a2017/ref/DiscretizingContinuousSystems.pdf
            # the full computation for discretization is fairly complicated and
            # involves calculating an integral of the matrix exponential
            self.Vd = self.V * Ts
            self.Wd = self.W / Ts
