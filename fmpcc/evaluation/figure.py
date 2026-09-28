import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def foresight(path, episodes, draw_constraints, limits, every, max_episodes=10, title=''):
    """The MPC foresight figure of DPCC: executed path (black) and the candidate plans (blue) every
    `every` control steps, one panel per episode. episodes: list of (path (T, 2), plans list of (B, H, 2))."""
    n = min(len(episodes), max_episodes)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, (executed, plans) in zip(axes[0], episodes[:n]):
        for k in range(0, len(plans), every):
            for cand in plans[k]:
                ax.plot(cand[:, 0], cand[:, 1], 'b', linewidth=0.8)
        executed = np.asarray(executed)
        ax.plot(executed[:, 0], executed[:, 1], 'k', linewidth=2)
        ax.plot(executed[0, 0], executed[0, 1], 'go')
        draw_constraints(ax)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_aspect('equal')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
