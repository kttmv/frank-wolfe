import numpy as np
from scipy.optimize import linprog, minimize_scalar
import numpy.testing as npt

def frank_wolfe(f, grad_f, constraints, x0, epsilon=1e-4, delta=1e-4, max_iter=100):
    x = x0
    i = 1
    while i < max_iter:
        # Step 1: Calculation of the function gradient at the current point
        grad = grad_f(x)
        if np.linalg.norm(grad) <= epsilon:
            break

        # Step 2: Solving the linear programming problem to find the direction
        lp_result = linprog(c=grad, A_ub=constraints[0], b_ub=constraints[1], bounds=[(0, None)] * len(x))
        y = lp_result.x

        # Step 3: Performing a linear search to determine the optimal step
        line_search_result = minimize_scalar(lambda a: f(x + a * (y - x)), bounds=(0, 1), method='bounded')
        a = line_search_result.x

        # Step 4: Updating the current point
        x_next = x + a * (y - x)

        # Step 5: Checking the stopping conditions
        if np.linalg.norm(x_next - x) < delta * np.linalg.norm(x_next) and abs(f(x_next) - f(x)) <= epsilon * abs(f(x_next)):
            break

        x = x_next
        i += 1


    return x, i

def print_result(result):
    print('--------')
    print('Solution:')
    for i, num in enumerate(result[0]):
        print(f'x_{i} = {num:.5f}')
    print()
    print(f'Number of iterations: {result[1]}')
    print('--------\n\n\n')



# EXAMPLE 1
def f1(x):
    return x[0]**2 + x[1]**2

def grad_f1(x):
    return np.array([2*x[0], 2*x[1]])

constraints1 = [np.array([[1, 1]]), np.array([1])]

x0_1 = np.array([0.5, -10])

result1 = frank_wolfe(f1, grad_f1, constraints1, x0_1)
print_result(result1)

# EXAMPLE 2
def f2(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def grad_f2(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 2)])

constraints2 = [np.array([[1, 2]]), np.array([2])]

x0_2 = np.array([0, 0])

result2 = frank_wolfe(f2, grad_f2, constraints2, x0_2)
print_result(result2)

# EXAMPLE 3
def f3(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def grad_f3(x):
    return np.array([2*x[0], 2*x[1], 2*x[2]])

constraints3 = [np.array([[1, 1, 1]]), np.array([1])]

x0_3 = np.array([1.5, 3.2, 0.5])

result3 = frank_wolfe(f3, grad_f3, constraints3, x0_3)
print_result(result3)


# EXAMPLE 4
def f(x):
    return -1 * (20 * np.sqrt(x[0]) + 30 * np.sqrt(x[1]))

def grad_f(x):
    return np.array([-10 / np.sqrt(x[0]), -15 / np.sqrt(x[1])])

constraints = [np.array([[1, 2], [2, 1]]), np.array([100, 160])]

x0 = np.array([1, 1])

result = frank_wolfe(f, grad_f, constraints, x0)
print_result(result)