
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib import animation, cm

# Parameters
A = 1.0
ALPHA = 0.33
DELTA = 0.1
SAVRATE = 0.2
RHO = 0.80
SIGMA = 0.05

# Option 1: Analytical deterministic (nontrivial) steady state
kss = (SAVRATE*A/DELTA)**(1.0/(1.0-ALPHA))
print("Nontrivial steady state, kss = " + str(kss))

# Option 2: Numerical: Solve for k s.t. g(k) := sk^alpha - dk = 0
# Here we use SCIPY/Brent's root finding algorithm
deepen_k = lambda x: SAVRATE*A*x**ALPHA
shallow_k = lambda x: DELTA*x
gk = lambda x: SAVRATE*x**ALPHA - DELTA*x

kmin = 0.0
kmax = kss*1.3
kss_numerical = opt.brentq(gk, kmin + 0.001, kmax,
                    args=(), xtol=1e-12,
                    rtol=4.4408920985006262e-12, maxiter=100,
                    full_output=True, disp=True)

# Check Options 1 and 2 produce same result
print("Analytical: Steady state per-capita capital, kss = " + str(kss))
print("Brent's Method: Steady state per-capita capital, kss_numerical = " + str(kss_numerical[0]))

# Define dicrete domain and function h:
k_set = np.linspace(kmin, kmax, 100)
h = lambda x: (1-DELTA)*x + SAVRATE*A*x**ALPHA

# Simulate equilibrium path
T = 200
k = np.zeros((T,1))
k[0] = 0.01
for t in range(T-1):
    k[t+1] = h(k[t])


# animation
# Setup: figure, axes, plot element to animate
fig0 = plt.figure(facecolor='white')
ax = plt.axes(xlim=(k_set.min(), k_set.max()), ylim=(k_set.min(), k_set.max()))
ax.plot(k_set, h(k_set), 'b-', # the recursive mapping h
         k_set, k_set, 'r--')   # 45-degree line
line, = ax.plot([], [], 'oy')
plt.xlabel('$k_{t}$')
plt.ylabel('$k_{t+1}$')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = k[i]
    y = h(k[i])
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anime = animation.FuncAnimation(fig0, animate, init_func=init,
                               frames=T, interval=200, blit=True)
anime.save('solowswan_anime.mp4', fps=10)
#anime.save('solowswan_anime.mp4', writer='ffmpeg', fps=10, \
#                                        xtra_args=['-vcodec', 'libx264'])

# Plot the phase diagram:
fig1 = plt.figure(facecolor='white')
plt.plot(k_set, h(k_set), 'b-', # the recursive mapping h
         k_set, k_set, 'r--',   # 45-degree line
         kss, kss, 'go')        # steady state point
plt.xlabel('$k_{t}$')
plt.ylabel('$k_{t+1}$')

# Plot the phase diagram:
fig2 = plt.figure(facecolor='white')
plt.plot(k_set, h(k_set), 'b-', # the recursive mapping h
         k_set, k_set, 'r--')   # 45-degree line
plt.plot(k[0:-2], k[1:-1], 'oy') # Equilibrium path
plt.plot(kss, kss, 'ro')        # steady state point
plt.xlabel('$k_{t}$')
plt.ylabel('$k_{t+1}$')

# Plot graphs of dk and sk^alpha
shallow = shallow_k(k_set)
deepen = deepen_k(k_set)
fig1 = plt.figure(facecolor='white')
plt.plot(k_set, shallow, 'r-',          # graph of deprecn function
         k_set, deepen, 'b-',           # graph of saving function
         kss, shallow_k(kss), 'go')     # steady state point
plt.xlabel('$k_{t}$')
plt.ylabel('saving, depreciation, output')

plt.show()
