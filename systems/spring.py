import numpy as np
from systems._system import SystemInterface

class Spring(SystemInterface):
    """
    Implementation of a spring-damper system with the form
    x'' = - c/m*x' - k/m * x + f(t)/m
    
    Attributes:
        k float: Spring Constant
        c float: Damping Constant
        m float: Mass of the Attached Object
    """

    def __init__(self, k=1E5, c=2E3, m=450):
        """
        Initialize the spring-damper system with a given set of parameters.

        Args:
            k: spring constant (N/m)
            c: damping coefficient (N·s/m)
            m: mass (kg)
        """
        self.k = k
        self.c = c
        self.m = m

        self.dw_sol = []

    def step(self, t, w, f= lambda t,w: 0):
        """
        Solution at a given time and state vector for the spring-damper system.

        Args:
        t: time (s)
        w: state vector [displacement (m), velocity (m/s)]
        f: external force function (N)

        Returns:
            np.array[float]: Derivative of the System        
        """

        # unpack state vector
        assert(np.array(w).size == 2)
        x, v, = w

        # compute derivatives
        dv = (-self.k*x - self.c*v + f(t,w))/self.m # Acceleration
        dx = v # Velocity

        dw = np.array([dx, dv])
        self.dw_sol.append(dw)

        return dw