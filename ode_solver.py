import numpy as np
import matplotlib.pyplot as plt
colors = ['b', 'g', 'r', 'c', 'm']

def ode_problem(y_vec):
    '''
    anharmonic oscillator
    '''
    y_0, y_1 = y_vec
    f_0 = y_1
    f_1 = -y_0 - (y_0)**3
    return np.array([f_0, f_1])

def rk4(ode_problem, a, b, n, init_vec):
    dt = (b-a)/(n-1)
    ts = a + np.arange(n) * dt
    chi_vecs = np.zeros((n, init_vec.size))
    chi_vec = np.copy(init_vec)
    for t_idx, t in enumerate(ts):
        chi_vecs[t_idx][:] = chi_vec
        k_1 = dt * ode_problem(chi_vec)
        k_2 = dt * ode_problem(chi_vec + (k_1)/2)
        k_3 = dt * ode_problem(chi_vec + (k_2)/2)
        k_3 = dt * ode_problem(chi_vec + k_3)
        chi_vec += (k_1 + 2*k_2 + 2*k_2 + k_3)/6
    return ts, chi_vecs

# phase flow (just for interest)
x_gebiet = np.arange(-5.2, 5.2, 0.01)
y_gebiet = np.arange(-20, 20, 0.01)
[X, Y] = np.meshgrid(x_gebiet, y_gebiet)
xdot = Y
ydot = -X - X**3
speed = np.sqrt(xdot**2+ydot**2)


# task parameters
a = 0
b = 20


'''
#### 5 displacements 
init_vecs = [np.array([1., 0]),  np.array([2., 0]), np.array([3., 0]), np.array([4., 0]), np.array([5., 0])]
fig, axes = plt.subplots(nrows=2)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})
for init_vec in init_vecs:
    xs, chi_vecs = rk4(ode_problem, a, b, 10000, init_vec)
    axes[0].plot(xs, chi_vecs[:, 0], label='$\chi(0)=${}'.format(init_vec[0]))
    axes[1].plot(chi_vecs[:, 0], chi_vecs[:, 1])
axes[0].legend(ncol=5,
                    fancybox=False,
                    edgecolor='k',
                    bbox_to_anchor=(0, 1),
                    loc=3)
axes[1].streamplot(X, Y, xdot, ydot, color=speed, cmap='gray_r', density=1.5)
axes[0].set_xlabel(r'$\tau$')
axes[0].tick_params(direction='in', bottom=True, left=True, right=True, top=True)
axes[0].set_ylabel('$\chi$')
axes[0].grid(True, alpha=0.5, linestyle=':', color='k')
axes[1].set_xlabel('$\chi$')
axes[1].tick_params(direction='in', bottom=True, left=True, right=True, top=True)
axes[1].set_ylabel('$\partial_t\chi$')
axes[1].grid(True, alpha=0.5, linestyle=':', color='k')
plt.show()
'''
#### hamiltonian
init_vecs = [np.array([1., 0]),  np.array([2., 0]), np.array([3., 0]), np.array([4., 0]), np.array([5., 0])]
fig, axes = plt.subplots(ncols=2)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})
for clr_idx, init_vec in enumerate(init_vecs):
    xs, chi_vecs = rk4(ode_problem, a, b, 10000, init_vec)
    axes[0].plot(xs, (1/2*chi_vecs[:, 0]**2 + 1/4*chi_vecs[:, 0]**4 + 1/2*chi_vecs[:, 1]**2)-((1/2*init_vec[0]**2 + 1/4*init_vec[0]**4 + 1/2*init_vec[1]**2)), label='$\chi(0)=${}'.format(init_vec[0]), color=colors[clr_idx])

for n in np.logspace(3, 5, 7).astype(int):
    for clr_idx, init_vec in enumerate(init_vecs):
        xs, chi_vecs = rk4(ode_problem, a, b, n, init_vec)
        axes[1].plot(n, (1/2*chi_vecs[-1, 0]**2 + 1/4*chi_vecs[-1, 0]**4 + 1/2*chi_vecs[-1, 1]**2)-((1/2*init_vec[0]**2 + 1/4*init_vec[0]**4 + 1/2*init_vec[1]**2)), marker='o', color=colors[clr_idx])

axes[0].legend(ncol=5,
                    fancybox=False,
                    edgecolor='k',
                    bbox_to_anchor=(0, 1),
                    loc=3)
axes[0].set_xlabel(r'$\tau$')
axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[0].tick_params(direction='in', bottom=True, left=True, right=True, top=True)
axes[0].set_ylabel('$H-H_{init}$')
axes[1].set_ylabel('$H_{end}-H_{init}$')
axes[0].grid(True, alpha=0.5, linestyle=':', color='k')
axes[1].set_xlabel('timesteps')
axes[1].tick_params(direction='in', bottom=True, left=True, right=True, top=True)
axes[1].grid(True, alpha=0.5, linestyle=':', color='k')
plt.show()


