import matplotlib.pyplot as plt
import numpy as np


def plot_max_statistic_distribution(statmap, statmap_null):
    alpha = 0.05

    nulltmax = statmap_null.max(axis=(0, 1))
    tmaxcutoff = np.quantile(nulltmax, 1 - alpha)

    obs_tmax = statmap.max()

    fig, ax = plt.subplots(figsize=(10, 4))

    # histogram
    n, bins, _ = ax.hist(
        nulltmax,
        bins=60,
        density=True,
        color='0.75',
        edgecolor='0.3',
        linewidth=0.5,
        label=r"$\max\ D^k_{\text{H0}}$"
    )

    # shade rejection region
    ax.axvspan(
        tmaxcutoff,
        bins[-1]+0.35,
        color='orange',
        alpha=0.2,
        label=r'$\alpha \geq 0.05$'
    )

    # critical value
    ax.axvline(
        tmaxcutoff,
        color='k',
        linestyle='--',
        linewidth=1.5,
        label=f'Critical value = {tmaxcutoff:.2f}'
    )

    # observed statistic
    ax.axvline(
        obs_tmax,
        color='red',
        linewidth=2,
        label=r'$\max\ D_{\text{obs}}$ = ' + f'{obs_tmax:.2f}'
    )

    ax.set_xlabel('Maximum test statistic', fontsize=15)
    ax.set_ylabel('Density', fontsize=15)
    ax.set_title('Raw Max-Statistic Test Distribution', fontsize=20, y=1.05)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 5)
    plt.xlim(0.5, 3.0)

    ax.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_statmap(statmap, drawidx=0, vm=1):
    if len(statmap.shape) == 2:
        statmap = statmap[:, :, np.newaxis]
    plt.imshow(statmap[:, :, drawidx], vmax=vm, vmin=-vm, cmap='seismic')
    plt.xticks([])
    plt.yticks([])
    plt.show()