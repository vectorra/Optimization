import matplotlib.pyplot as plt
from annealing import SA_optimize
from geometry import compute_perimeter
from plotting import plot_state

# Define crate dimensions: (id, width, height)
crates = [
    (1, 3, 4),
    (2, 5, 2),
    (3, 4, 4),
    (4, 6, 1),
    (5, 2, 7),
    (6, 3, 3),
    (7, 5, 5),
    (8, 2, 6),
    (9, 3, 2),
    (10, 4, 3)
]

boundaries = [20, 20]

# Run optimization
history, best_state_end, perimeters, best_perimeters = SA_optimize(crates, boundaries)

# Final layout
fig, ax = plt.subplots(figsize=(6, 6))
colors = plt.cm.tab10.colors
plot_state(best_state_end, "Best", ax, colors, boundaries)
ax.set_title(f"Best perimeter = {compute_perimeter(best_state_end):.2f}")
plt.tight_layout()
plt.show(block=True)

# Plot metrics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(perimeters, color='dodgerblue')
ax1.set_title("Objective at each iteration")
ax1.set_ylabel("Objective")
ax1.grid(True)

ax2.plot(best_perimeters, color='deepskyblue')
ax2.set_title("Best known solution")
ax2.set_ylabel("Best Objective")
ax2.set_xlabel("Iteration")
ax2.grid(True)

plt.tight_layout()
plt.show(block=True)
