import numpy as np
import sys
import quadprog
import scipy.linalg

# for the input integrator formulation, please refer to
# https://www.kth.se/social/upload/5194b53af276547cb18f4624/lec13_mpc2_4up.pdf

def MPC(model, xk, x_ref, Q, R, use_input_integrator = False):
    # length of the prediction horizon
    L = int(round(x_ref.shape[0] / xk.shape[0]))

    # number of states
    N = model.x0.shape[0]

    # number of inputs
    M = model.u0.shape[0]

    # transform current state and reference trajectory relative to linearization
    # point
    xk_t = np.copy(xk)
    x_ref_t = np.copy(x_ref)
    if (use_input_integrator):
        xk_t = xk - np.vstack([model.x0, np.zeros((1,M))])
    else:
        xk_t = xk - model.x0

    for i in range(0, L):
        if (use_input_integrator):
            x_ref_t[i*(N+M):(i+1)*(N+M)] = x_ref_t[i*(N+M):(i+1)*(N+M)] - np.vstack([model.x0, np.zeros((1,M))])
        else:
            x_ref_t[i*N:(i+1)*N] = x_ref_t[i*N:(i+1)*N] - model.x0

    if (use_input_integrator):
        # form new state space representation
        Ad = np.block([[model.Ad, model.Bd], [np.zeros((M, N)), np.eye(M, M)]])
        Bd = np.block([[np.zeros((N, 1))], [np.ones((M, 1))]])

        # for the input integrator formulation, we augment the state by one for
        # every input to the system
        N = N + M
    else:
        Ad = model.Ad
        Bd = model.Bd

    # pre-compute powers of the A matrix for efficiency
    pow_a = [np.identity(N)]
    for i in range(1, L+1):
        pow_a.append(pow_a[i-1] @ Ad)

    A = np.empty((N*L, N))
    for i in range(0, L):
        A[i*N:(i+1)*N, 0:N] = pow_a[i+1]

    B = np.zeros((N*L, M*L))
    B[0:N, 0:M] = Bd
    for i in range(1, L):
        for j in range(0, i+1):
            if (j == 0):
                B[i*N:(i+1)*N, j*M:(j+1)*M] = pow_a[i] @ Bd
            else:
                B[i*N:(i+1)*N, j*M:(j+1)*M] = B[(i-1)*N:i*N, (j-1)*M:j*M]

    Q_hat = np.kron(np.eye(L+1), Q)
    R_hat = np.kron(np.eye(L), R)

    Q_hat = np.kron(np.eye(L), Q)
    R_hat = np.kron(np.eye(L), R)

    # determine the optimal terminal cost matrix that stabilizes MPC, see
    # https://www.mathworks.com/help/mpc/ug/terminal-weights-and-constraints.html
    Qf = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    Q_hat[(L-1)*N:,(L-1)*N:] = Qf

    H = B.transpose() @ Q_hat @ B + R_hat
    f = -(xk_t.transpose() @ A.transpose() @ Q_hat @ B - x_ref_t.transpose() @ Q_hat @ B)

    u = np.array([quadprog.solve_qp(H, f.flatten())[0][0:M]])
    u = u + xk_t[N-M:, 0]
    return u
