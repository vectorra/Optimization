import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import grad
from matplotlib.animation import FuncAnimation


# --- Objective Function and Gradient ---
def himmelblau(x):
    return (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2

objective = himmelblau
objective_grad = grad(objective)


def visualize(optimizers, start_point=None, max_iters=250):
    if start_point is None:
        start_point = np.array([-2.0, 2.0])

    xlist = np.linspace(-5, 5, 400)
    ylist = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.vectorize(lambda x, y: objective(anp.array([x, y])))(X, Y)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-10, cmap='viridis')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(np.min(Z)-10, np.max(Z)+10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('Optimization Trajectories on Himmelblau Function')

    colors = ['crimson', 'dodgerblue', 'darkorange', 'green',
            'magenta', 'teal', 'black', 'slateblue']

    markers = ['o', '^', 's', 'D', '*', 'P', 'x', 'v']

    linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']

    marker_sizes = [8, 9, 10, 8, 9, 8, 8, 9]

    line_widths = [3, 3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]


    trajectories = []
    points = []
    lines = []

    trail_len = 15

    for i, (name, opt) in enumerate(optimizers.items()):
        print("name", name)
        pos = start_point.copy()
        trajectories.append([pos.copy()])
        p, = ax.plot([pos[0]], [pos[1]], [objective(pos)],
                    marker=markers[i], color=colors[i],
                    label=name, markersize=marker_sizes[i], linestyle='None')
        points.append(p)

        l, = ax.plot([pos[0]], [pos[1]], [objective(pos)],
                    color=colors[i], lw=line_widths[i], linestyle=linestyles[i])
        lines.append(l)

    ax.legend(fontsize=12)

    def update(frame):
        for i, (name, opt) in enumerate(optimizers.items()):
            pos = trajectories[i][-1]
            grad_val = objective_grad(pos)
            pos_new = opt.step(pos, grad_val)
            trajectories[i].append(pos_new)

            trail_points = trajectories[i][-trail_len:]
            data = np.array(trail_points)
            zdata = np.array([objective(p) for p in data])

            points[i].set_data([data[-1, 0]], [data[-1, 1]])
            points[i].set_3d_properties([zdata[-1]])
            lines[i].set_data(data[:, 0], data[:, 1])
            lines[i].set_3d_properties(zdata)

        return points + lines

    anim = FuncAnimation(fig, update, frames=max_iters, interval=50, blit=False)

    def reset_animation(event):
        if event.key == 'r':
            print("Restarting animation")
            for traj in trajectories:
                traj.clear()
                traj.append(start_point.copy())
            for opt in optimizers.values():
                if hasattr(opt, 'reset'):
                    opt.reset()
            anim.frame_seq = anim.new_frame_seq()

    fig.canvas.mpl_connect('key_press_event', reset_animation)

    plt.show()

