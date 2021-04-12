import numpy as np
import matplotlib.pyplot as plt

# data given by exercise
x_min = 0.
x_max = 2.
x_true = np.pi/4.
tol = np.finfo(float).eps              # smallest absolute float number

def test_function(x):
    res = 1/np.sqrt(2) - np.cos(x)
    return res

def test_derivative(x):
    res = np.sin(x)
    return res

# plotting
def plot_errors(bi_err, new_err, sec_err):
    colors = ['#e40017', '#1F618D', '#5b6d5b', '#484018']
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
    })
    fig, axes = plt.subplots(figsize=(3.3, 2.5))
    axes.tick_params(direction='in', bottom=True, left=True, right=True, top=True)
    axes.set_yscale('log')
    axes.grid(True, alpha=0.5, linestyle=':', color='k')
    axes.plot(np.arange(len(bi_err)) + 1, bi_err, color=colors[0], label='bisection')
    axes.plot(np.arange(len(new_err)) + 1, new_err, color=colors[1], label='Newton-Raphson')
    axes.plot(np.arange(len(sec_err)) + 1, sec_err, color=colors[3], label='secant')
    axes.plot(6, new_err[-1], marker='.', color=colors[1])
    axes.plot(8, sec_err[-1], marker='.', color=colors[3])
    axes.legend(ncol=3, fancybox=False, edgecolor='k', bbox_to_anchor=(0, 1), loc=3)
    axes.set_xlabel('iteration')
    axes.set_ylabel('absolute error')
    plt.show()


# methods
def bisection_until_error(f, error=tol):
    x_zero = 0.
    x_l = x_min
    x_b = x_max
    n = 0
    error_list = []
    while True:
        n += 1
        x_mid = (x_l + x_b) / 2.
        error_list.append(np.abs(x_mid - x_true))
        if np.abs(f(x_zero)) <= error or (x_b-x_l)/2 <= 2*error:
            x_zero = x_mid
            break
        else:
            if f(x_l)*f(x_mid) < 0:
                x_l = x_l
                x_b = x_mid
            else:
                x_l = x_mid
                x_b = x_b
    return x_zero, error_list

def newton_raphson_until_error(f, df_dx, error = tol):
    x_zero = 0.2
    n = 0
    error_list = []
    while np.abs(x_zero - x_true) > error:
        n += 1
        if f(x_zero) / df_dx(x_zero) == 0:
            break                          # fixed-point criteria
        else:
            x_zero = x_zero - f(x_zero) / df_dx(x_zero)
            error_list.append(np.abs(x_zero - x_true))
    return x_zero, error_list

def secant_until_error(f, error = tol):
    x_zero = 0.
    x_values = [2., 0.]                    # que
    n = 0
    error_list = []
    while np.abs(x_zero-x_true) > error:
        n += 1
        if f(x_values[0]) * (x_values[0]-x_values[1]) / (f(x_values[0]) - f(x_values[1])) == 0:
            break                          # fixed-point criteria
        else:
            x_zero = x_values[0] - (f(x_values[0]) * (x_values[0]-x_values[1])) / (f(x_values[0]) - f(x_values[1]))
            error_list.append(np.abs(x_zero - x_true))
            x_values.pop()
            x_values.insert(0, x_zero)     # mimic que behavior
    return x_zero, error_list

# script
bi_x_zero, bi_err_list = bisection_until_error(f=test_function)
new_x_zero, new_err_list = newton_raphson_until_error(f=test_function, df_dx=test_derivative)
sec_x_zero, sec_err_list = secant_until_error(f=test_function)
plot_errors(bi_err=bi_err_list, new_err=new_err_list, sec_err=sec_err_list)