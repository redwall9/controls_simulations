import numpy as np
import sys
import quadprog
import scipy.linalg

def MPC(model, xk, x_ref, Q, R):
    # length of the prediction horizon
    L = int(round(x_ref.shape[0] / xk.shape[0]))

    # number of states
    N = model.x0.shape[0]

    # number of inputs
    M = model.u0.shape[0]

    # transform current state and reference trajectory relative to linearization
    # point
    xk_t = xk - model.x0
    x_ref_t = np.copy(x_ref)
    for i in range(0, L):
        x_ref_t[i*N:(i+1)*N] = x_ref_t[i*N:(i+1)*N] - model.x0

    # pre-compute powers of the A matrix for efficiency
    pow_a = [np.identity(N)]
    for i in range(1, L+1):
        pow_a.append(pow_a[i-1] @ model.Ad)

    A = np.empty((N*L, N))
    for i in range(0, L):
        A[i*N:(i+1)*N, 0:N] = pow_a[i+1]

    B = np.zeros((N*L, M*L))
    B[0:N, 0:M] = model.Bd
    for i in range(1, L):
        for j in range(0, i+1):
            if (j == 0):
                B[i*N:(i+1)*N, j*M:(j+1)*M] = pow_a[i] @ model.Bd
            else:
                B[i*N:(i+1)*N, j*M:(j+1)*M] = B[(i-1)*N:i*N, (j-1)*M:j*M]

    Q_hat = np.kron(np.eye(L+1), Q)
    R_hat = np.kron(np.eye(L), R)

    Q_hat = np.kron(np.eye(L), Q)
    R_hat = np.kron(np.eye(L), R)

    # determine the optimal terminal cost matrix that stabilizes MPC, see
    # https://www.mathworks.com/help/mpc/ug/terminal-weights-and-constraints.html
    Qf = scipy.linalg.solve_discrete_are(model.Ad, model.Bd, Q, R)
    Q_hat[(L-1)*N:,(L-1)*N:] = Qf

    H = B.transpose() @ Q_hat @ B + R_hat
    f = -(xk_t.transpose() @ A.transpose() @ Q_hat @ B - x_ref_t.transpose() @ Q_hat @ B)

    return quadprog.solve_qp(H, f.flatten())[0]