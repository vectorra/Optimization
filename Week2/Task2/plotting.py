import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

def plot_state(state, step_num, ax, color_map, boundaries):
    for i, (id, b) in enumerate(state):
        x_min, y_min, x_max, y_max = b.bounds
        w = x_max - x_min
        h = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), w, h,
                                 linewidth=1, edgecolor='black',
                                 facecolor=color_map[i % len(color_map)], alpha=0.6)
        ax.add_patch(rect)
        ax.text(x_min + w/2, y_min + h/2, str(id),
                ha='center', va='center', fontsize=10, weight='bold')
    ax.set_title(f"Step {step_num}")
    ax.set_xticks(range(0, int(boundaries[0]) + 1))
    ax.set_yticks(range(0, int(boundaries[1]) + 1))
    ax.grid(True, which='both')
    ax.set_aspect('equal')
    ax.grid(True)

def initialize_live_plot(boundaries):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10.colors
    ax.set_xlim(0, boundaries[0])
    ax.set_ylim(0, boundaries[1])
    ax.set_aspect('equal')
    ax.grid(True)
    return fig, ax, colors

def animate_solution(history, boundaries, interval=300):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10.colors
    def init():
        ax.clear()
        ax.set_xlim(0, boundaries[0])
        ax.set_ylim(0, boundaries[1])
        ax.set_aspect('equal')
        ax.grid(True)
        return []
    def update(step):
        ax.clear()
        plot_state(history[step], step, ax, colors, boundaries)
        return []
    ani = animation.FuncAnimation(
        fig, update, frames=len(history), init_func=init,
        interval=interval, blit=False, repeat=False
    )
    plt.show()
