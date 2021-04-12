import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
tol = np.finfo(float).eps

def x_t(t, t_b):
    res = -np.cos(t) + np.cos(t_b) - np.sin(t_b)*(t - t_b)
    return res

def v_t(t, t_b):
    res = np.sin(t) - np.sin(t_b)
    return res

def energy(t, t_b):
    res = (v_t(t, t_b)**2)/2
    return res

def bisection(t_b, n=300, f=x_t, t_left=0. + tol, t_great=2*np.pi + tol, error = tol):
    t_l = t_left
    t_g = t_great
    t_zero = False
    i = 0
    while i < n:
        i += 1
        t_r = (t_l + t_g) / 2.
        if np.abs(f(t_r, t_b)) <= 2*error:
            t_zero = t_r
            break
        else:
            if f(t_l, t_b)*f(t_r, t_b) < 0:
                t_g = t_r
            else:
                t_l = t_r
    return t_zero

def bisection_with_arrays(arr, n = 300, error=tol):
    i_l = 0
    i_g = len(arr) - 1
    i_zero = False
    m = 0
    while m < n:
        m += 1
        i_r = (i_l + i_g) // 2
        if i_g - i_l == 1:
            return i_r
        elif m == n-1:
            i_zero = i_r
        else:
            if arr[i_l] * arr[i_r] < 0:
                i_g = i_r
            else:
                i_l = i_r
    return i_zero

def decide_recollision(t_b):
    if bisection(t_b) == False:
        return False
    else:
        return True

def finite_difference(arr, t_b_left, t_b_right, n):
    dt = (t_b_right-t_b_left)/n
    derivative = np.array([j-i for i, j in zip(arr[:-1], arr[1:])]) / dt
    index = bisection_with_arrays(derivative)
    t_b_max = t_b_left + dt*index  # reconstruct index to time
    return t_b_max

def get_table(t_b_zeroes, recollision_times):
    energies = []
    for elem_t_b, elem_t_r in zip(t_b_zeroes, recollision_times):
        energies.append(energy(elem_t_r, elem_t_b))
    df = pd.DataFrame(dict(birth_times=t_b_zeroes,
                           recollision_times=recollision_times,
                           recollision_energy=np.array(energies)*4)     # U_p units
                      )
    print(df.to_latex(index=False))

def get_recollision_times(f, birth_times):
    t_r = np.zeros(len(birth_times))
    for elem, index in zip(birth_times, range(len(birth_times))):
        t_r[index] = bisection(f=f, t_b=birth_times[index])
    return t_r

def get_recollision_energies(t_b_left, t_b_right, n):
    t_b_list = np.linspace(t_b_left, t_b_right, n)
    t_b_filtered = list(filter(decide_recollision, t_b_list))
    t_r_list = get_recollision_times(x_t, t_b_filtered)
    recoll_energies = [energy(t_r, t_b)*4 for t_r, t_b in zip(t_r_list, t_b_filtered)] # * 4 in order to get U_p scale
    return recoll_energies

def plot_phase_space(t_b, t_b_zeroes, recollision_times):
    t_general = np.linspace(-1, 10, 100)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
    })
    fig, axes = plt.subplots(1, 2)
    axes[0].tick_params(direction='in', bottom=True, left=True, right=True, top=True)
    axes[0].grid(True, alpha=0.5, linestyle=':', color='k')
    axes[0].set_xlabel('$t$')
    axes[0].set_ylabel('$x(t;t_b)$')
    for elem in t_b:
        if elem >= 0:
            axes[0].plot(t_general, x_t(t_general, elem))
        else:
            axes[0].plot(t_general, x_t(t_general, elem), color='k', linestyle=':')
    axes[1].tick_params(direction='in', bottom=True, left=True, right=True, top=True)
    for elem_tb, elem_recoll in zip(t_b_zeroes, recollision_times):
        t = np.linspace(elem_tb, elem_recoll, 100)
        axes[1].plot(x_t(t, t_b=elem_tb), v_t(t, t_b=elem_tb), label='$t_b=${}'.format(elem_tb))
    axes[1].legend(ncol=3,
                    fancybox=False,
                    edgecolor='k',
                    bbox_to_anchor=(-1, 1),
                    loc=3)
    axes[1].grid(True, alpha=0.5, linestyle=':', color='k')
    axes[1].set_xlabel('$x(t;t_b)$')
    axes[1].set_ylabel('$v(t;t_b)$')
    plt.show()

def plot_all_results(t_b_filtered, t_recollisions, t_b_max, t_r_max, recoll_energies):
    fig, axes = plt.subplots()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
    })
    # plot for E-field
    t = np.linspace(0, 2 * np.pi, 200)
    axes.plot(t, -np.cos(t), color='k', label='laser field')
    axes.tick_params(direction='in', bottom=True, left=True, right=True, top=True)
    axes.grid(True, alpha=0.5, linestyle=':', color='k')
    # plots for parametric trajectories
    for elem_tb, elem_recoll in zip(t_b_filtered, t_recollisions):
        t = np.linspace(elem_tb, elem_recoll, 200)
        axes.plot(t, x_t(t, elem_tb), color='r', linestyle='-.')
    # plot for max energy trajectory
    t_max = np.linspace(t_b_max, t_r_max, 200)
    axes.plot(t_max, x_t(t_max, t_b_max), color='r', linewidth=2, label='$x(t;t_b(E_{max}))$')
    # plot for energy
    t_energies = np.linspace(0, np.pi/2, len(recoll_energies))
    axes.plot(t_energies, recoll_energies, label='$E_{coll} [U_p]$')
    axes.legend(ncol=1, fancybox=False, edgecolor='k')
    axes.set_xlabel('time / birth time')
    plt.show()

# script
t_b_exercise = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
t_b_filtered = list(filter(decide_recollision, t_b_exercise))  # show recollision behavior
recollision_times = get_recollision_times(x_t, t_b_filtered)

# nice plot to see overall behavior
# plot_phase_space(t_b_exercise, t_b_filtered, recollision_times=recollision_times)

# automatic table for latex
# get_table(t_b_filtered, recollision_times)


# number of segments = 1000
n = 1000

# first interval:
recoll_energies = get_recollision_energies(0, np.pi/2, n) # full half cycle
print(finite_difference(recoll_energies, 0, np.pi/2, n))

# second interval:
recoll_energies_2 = get_recollision_energies(0.25, 0.35, n)
print(finite_difference(recoll_energies_2, 0.25, 0.35, n))

# third interval:
recoll_energies_3 = get_recollision_energies(0.31, 0.32, n)
print(finite_difference(recoll_energies_3, 0.31, 0.32, n))

# fourth interval:
recoll_energies_4 = get_recollision_energies(0.312, 0.315, n)
print(finite_difference(recoll_energies_4, 0.312, 0.315, n))

# fifth interval:
recoll_energies_5 = get_recollision_energies(0.3133, 0.3134, n)
t_b_max = finite_difference(recoll_energies_5, 0.3133, 0.3134, n)
print(t_b_max)

# all results combined
t_max_recoll = get_recollision_times(x_t, [t_b_max])
E_max = energy(t_max_recoll, t_b=t_b_max) * 4 # * 4 to get E in U_p units
print(E_max)
plot_all_results(t_b_filtered=t_b_filtered,
                 t_recollisions=recollision_times,
                 t_b_max=t_b_max,
                 t_r_max=t_max_recoll,
                 recoll_energies=recoll_energies
                 )
