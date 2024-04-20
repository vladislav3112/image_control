import numpy as np
from scipy.integrate import odeint
import math

g = 10
L = 0.5
q0 = 0.5 # starting position
dt = 1/240 # time step
t = 0
maxTime = 5
logTime = np.arange(0.0, maxTime, dt)
sz = logTime.size

def rp(x, t):
    return [x[1], 
            -g/L*math.sin(x[0])]

theta = odeint(rp, [q0, 0], logTime)
logTheta = theta[:,0]

import matplotlib.pyplot as plt
plt.grid(True)
plt.plot(logTime, logTheta, label = "theor")
plt.legend()
plt.show()