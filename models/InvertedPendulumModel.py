# This class is a concrete implementation of the model class for an inverted
# pendulum. This model is described in
# https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
# The primary difference is that for this model the full nonlinear system
# dynamics are used for simulation, and instead of utilizing the small angle
# approximation the standard Jacobian linearization method is used. This results
# in a significantly more complicated system, but one that is a little more
# accurate. Calculating the system equations by hand is nearly impossible
# without making an algebraic error, which is why sympy was used to perform the
# derivation symbolically; see the derive_model_symbolically function. It takes
# the two nonlinear equations of motion, substitutes them into one another to
# get equations for x_ddot and phi_ddot in terms of only the state variables,
# and then computes the jacobians of those functions and prints everything out.
# The printed equations are (almost) valid Python code, so they can simply be
# copy and pasted into the appropriate places in get_model_derivative and the
# compute_* functions in the InvertedPendulumModel class.

from controls.Model import Model
import numpy as np
import scipy.signal
import math
import sympy

class InvertedPendulumModel(Model):
    # M: mass of the cart
    # m: mass of the pendulum
    # b: coefficient of friction of the cart
    # L: length of the pendulum
    # I: moment of inertia of the pendulum
    def __init__(self, M, m, b, L, I, x0, u0, V, W, Ts):
        self.M = M
        self.m = m
        self.b = b
        self.L = L
        self.I = I
        self.g = 9.807
        super().__init__(x0, u0, V, W, Ts)

    def get_model_derivative(self, x, u):
        M = self.M
        m = self.m
        b = self.b
        L = self.L
        I = self.I
        g = self.g
        x_dot = x[1][0]
        phi = x[2][0]
        phi_dot = x[3][0]

        return np.array([
            [x_dot],
            [(-I*L*m*phi_dot**2*math.sin(phi) - I*b*x_dot + I*u[0][0] - L**3*m**2*phi_dot**2*math.sin(phi) - L**2*b*m*x_dot + L**2*g*m**2*math.sin(2*phi)/2 + L**2*m*u[0][0])/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)],
            [phi_dot],
            [L*m*(-L*m*phi_dot**2*math.sin(2*phi)/2 + M*g*math.sin(phi) - b*x_dot*math.cos(phi) + g*m*math.sin(phi) + u[0][0]*math.cos(phi))/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)],
        ])

    def get_output(self, x):
        # the outputs of this system are the cart position and angle of
        # deviation, which can be measured by encoders in a real system
        return np.array([
            [x[0][0]],
            [x[2][0]]
        ])

    def compute_df_dx(self, x, u):
        M = self.M
        m = self.m
        b = self.b
        L = self.L
        I = self.I
        g = self.g
        x_dot = x[1][0]
        phi = x[2][0]
        phi_dot = x[3][0]

        return np.array([
            [0, 1, 0, 0],
            [0, (-I*b - L**2*b*m)/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2), -2*L**2*m**2*(-I*L*m*phi_dot**2*math.sin(phi) - I*b*x_dot + I*u[0][0] - L**3*m**2*phi_dot**2*math.sin(phi) - L**2*b*m*x_dot + L**2*g*m**2*math.sin(2*phi)/2 + L**2*m*u[0][0])*math.sin(phi)*math.cos(phi)/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)**2 + (-I*L*m*phi_dot**2*math.cos(phi) - L**3*m**2*phi_dot**2*math.cos(phi) + L**2*g*m**2*math.cos(2*phi))/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2), (-2*I*L*m*phi_dot*math.sin(phi) - 2*L**3*m**2*phi_dot*math.sin(phi))/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)],
            [0, 0, 0, 1],
            [0, -L*b*m*math.cos(phi)/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2), -2*L**3*m**3*(-L*m*phi_dot**2*math.sin(2*phi)/2 + M*g*math.sin(phi) - b*x_dot*math.cos(phi) + g*m*math.sin(phi) + u[0][0]*math.cos(phi))*math.sin(phi)*math.cos(phi)/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)**2 + L*m*(-L*m*phi_dot**2*math.cos(2*phi) + M*g*math.cos(phi) + b*x_dot*math.sin(phi) + g*m*math.cos(phi) - u[0][0]*math.sin(phi))/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2), -L**2*m**2*phi_dot*math.sin(2*phi)/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)],
        ])

    def compute_df_du(self, x, u):
        M = self.M
        m = self.m
        L = self.L
        I = self.I
        phi = x[2][0]

        return np.array([
            [0],
            [(I + L**2*m)/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)],
            [0],
            [L*m*math.cos(phi)/(I*M + I*m + L**2*M*m + L**2*m**2*math.sin(phi)**2)]
        ])

    def compute_dh_dx(self, x, u):
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

    def linearize_and_discretize_model_internal(self, x0, u0, Ts):
        self.Ad = self.compute_df_dx(x0, u0)
        self.Bd = self.compute_df_du(x0, u0)
        self.Cd = self.compute_dh_dx(x0, u0)

        D = np.array([
            [0],
            [0]
        ])

        self.Ad, self.Bd, self.Cd, D, dt = scipy.signal.cont2discrete((self.Ad, self.Bd, self.Cd, D), Ts, method='zoh')

        self.Ad = np.array(self.Ad)
        self.Bd = np.array(self.Bd)
        self.Cd = np.array(self.Cd)

    def derive_model_symbolically():
        M, m, b, L, g, u, I, phi, phi_dot, phi_ddot, x, x_dot, x_ddot = sympy.symbols('M m b L g u I phi phi_dot phi_ddot x x_dot x_ddot')

        eqn1 = sympy.Eq((M+m)*x_ddot + b*x_dot - m*L*phi_ddot*sympy.cos(phi) + m*L*phi_dot*phi_dot*sympy.sin(phi), u)
        eqn2 = sympy.Eq((I+m*L*L)*phi_ddot, m*L*(g*sympy.sin(phi) + x_ddot*sympy.cos(phi)))

        eqn1_x_ddot = sympy.solve(eqn1, x_ddot)[0]
        eqn2_phi_ddot = sympy.solve(eqn2, phi_ddot)[0]

        expr_x_ddot = sympy.solve(eqn1.subs(phi_ddot, eqn2_phi_ddot), x_ddot)[0]
        expr_phi_ddot = sympy.solve(eqn2.subs(x_ddot, eqn1_x_ddot), phi_ddot)[0]

        print(expr_x_ddot)
        print(expr_phi_ddot)

        print(sympy.Matrix([expr_x_ddot]).jacobian([x, x_dot, phi, phi_dot]))
        print(sympy.Matrix([expr_phi_ddot]).jacobian([x, x_dot, phi, phi_dot]))
        print(sympy.Matrix([expr_x_ddot]).jacobian([u]))
        print(sympy.Matrix([expr_phi_ddot]).jacobian([u]))

