import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Physics Constants
G = 1.0
softening = 0.02  # Crucial: prevents LSODA/DOP853 "t+h=t" errors

def derivatives(t, state):
    # state: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
    r = state[:6].reshape((3, 2))
    v = state[6:]
    dvdt = np.zeros((3, 2))
    
    for i in range(3):
        for j in range(3):
            if i != j:
                diff = r[j] - r[i]
                dist = np.linalg.norm(diff)
                # Softened Gravity Formula
                dvdt[i] += G * diff / (dist**2 + softening**2)**1.5
    
    return np.concatenate([v, dvdt.flatten()])

# 2. Initial Conditions (Figure-8 Stable Orbit)
state0 = [
    0.97000436, -0.24308753,  # Pos 1
    -0.97000436, 0.24308753,  # Pos 2
    0.0, 0.0,                 # Pos 3
    0.46620368, 0.43236573,   # Vel 1
    0.46620368, 0.43236573,   # Vel 2
    -0.93240737, -0.86473146  # Vel 3
]

# 3. Solve the ODE
t_eval = np.linspace(0, 10, 800)
sol = solve_ivp(derivatives, (0, 10), state0, t_eval=t_eval, method='DOP853')

# 4. Visualization Loop
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor('black')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

colors = ['#FF5733', '#33FF57', '#3357FF']
lines = [ax.plot([], [], color=colors[i], alpha=0.5, lw=1)[0] for i in range(3)]
dots = [ax.plot([], [], 'o', color=colors[i], ms=8)[0] for i in range(3)]

print("Starting animation...")
for f in range(len(t_eval)):
    for i in range(3):
        # Update trails (last 30 steps)
        start = max(0, f - 30)
        lines[i].set_data(sol.y[2*i, start:f], sol.y[2*i+1, start:f])
        # Update current point
        dots[i].set_data([sol.y[2*i, f]], [sol.y[2*i+1, f]])
    
    plt.pause(0.001)  # Forces the UI to refresh each frame

plt.ioff()
plt.show()