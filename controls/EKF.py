import numpy as np
import copy

# for now, this class assumes that you use a constant sampling rate, and that
# the prediction and update steps happen at the same time. As a result, it also
# assumes that all sensors update at the same rate. A future extension is to
# make the implementation support asynchronous sensor updates with variable
# sampling rates, which represents reality much better.

class EKF():
    def __init__(self, model, x0, P0):
        self.model = copy.deepcopy(model)
        self.x_hat = np.copy(x0)
        self.P = np.copy(P0)

    def predict_and_update(self, u, z):
        self.x_hat = self.x_hat +                                               \
                self.model.get_model_derivative(self.x_hat, u) * self.model.Ts

        A = self.model.compute_df_dx(self.x_hat, u)
        C = self.model.compute_dh_dx(self.x_hat, u)

        Pdot = A @ self.P + self.P @ A.transpose() + self.model.V
        self.P = self.P + Pdot*self.model.Ts

        K = self.P @ C.transpose() @ np.linalg.inv(C @ self.P @ C.transpose() + self.model.W)

        self.x_hat = self.x_hat + K @ (z - self.model.get_output(self.x_hat))
        self.P = (np.identity(self.P.shape[0]) - K @ C) @ self.P