import math
import copy
import random
from neighbours import neighbour
from geometry import compute_perimeter, random_initial_state
from plotting import initialize_live_plot

T0 = 100
cooling_rate = 0.98
target = 46
tolerance = 1e-1

def t_log_schedule(k, t1=100.0):
    if k == 0:
        return t1
    return t1 * math.log(2) / math.log(k + 1)

def loss(best_it, neighbour_state):
    neighbour_perimeter = compute_perimeter(neighbour_state)
    return neighbour_perimeter - best_it

def SA_optimize(crates, boundaries):
    x = random_initial_state(crates, boundaries)
    perimeters = []
    best_perimeters = []
    best_it = compute_perimeter(x)
    best_state = copy.deepcopy(x)
    T = T0
    step = 0
    fig, ax, colors = initialize_live_plot(boundaries)
    history = [x] 
    while True:
        x_new = None
        for _ in range(10): 
            candidate = neighbour(x, boundaries, best_state, T)
            if candidate is not None and candidate != x:
                x_new = candidate
                break
        if x_new is None:
            T = t_log_schedule(step)
            step += 1
            continue

        if not step % 1000:
            print("Step: ", step, " T: ", T)
        lost = loss(best_it, x_new)
        if lost < 0:
            x = x_new
            history.append(x)
            best_it = compute_perimeter(x_new)
            best_state = x_new.copy()
            perimeters.append(compute_perimeter(x))
            best_perimeters.append(best_it)
            print(f"🎯 At step {step}: perimeter = {compute_perimeter(x):.2f}")

        epsilon = 1e-8
        p = math.exp(-lost / (T+epsilon)) if lost >= 0 else 1.0
        accept = random.random() < p
        if accept:
            x = x_new
            history.append(x)
            perimeters.append(compute_perimeter(x))
            best_perimeters.append(best_it)

        if compute_perimeter(x) <= target + tolerance:
            print(f"🎯 Early stopping at step {step}: perimeter = {compute_perimeter(x):.2f}")
            break

        step += 1

    return history, best_state, perimeters, best_perimeters
