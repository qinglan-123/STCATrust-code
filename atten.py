import numpy as np
import matplotlib.pyplot as plt

# Set global font and style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10
})

# Create canvas and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
plt.subplots_adjust(wspace=0.3)


def plot_dataset(ax, attention, interaction, categories, title):
    """ Plot dual-axis combination chart for a single dataset """
    timeslots = np.arange(0, len(attention))  # Dynamic timeslots based on input data
    
    # Main axis (left) settings
    ax.set_xlim(-0.5, len(attention)-0.5)
    ax.set_xticks(timeslots)
    ax.set_xlabel("Timeslot", labelpad=10)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Attention Scores", labelpad=10)

    # Plot main attention score line
    main_line = ax.plot(timeslots, attention,
                        color='#1f77b4', marker='x', markersize=8,
                        linewidth=1.5, linestyle='--', label='Attention Scores')

    # Plot category attention scores
    colors = ['#ff7f0e', '#2ca02c']
    category_lines = []
    for i, (label, values) in enumerate(categories.items()):
        line = ax.plot(timeslots, values,
                       color=colors[i], marker='o' if i == 0 else 's',
                       markersize=6, linewidth=1.2, linestyle='-',
                       label=label)
        category_lines.append(line[0])

    # Secondary axis (right) settings
    ax2 = ax.twinx()
    ax2.set_ylim(0, max(interaction)*1.1 if len(interaction) > 0 else 1000)
    ax2.set_ylabel("# Interactions", labelpad=10)
    ax2.set_yticks(np.linspace(0, max(interaction)*1.1 if len(interaction) > 0 else 1000, 6))

    # Plot interaction count bars
    bars = ax2.bar(timeslots, interaction,
                   width=0.6, alpha=0.7,
                   color='#7f7f7f', edgecolor='k',
                   linewidth=0.5, label='# Interactions')

    # Combine legend
    lines = [main_line[0]] + category_lines + [bars]
    labels = ['Attention Scores'] + list(categories.keys()) + ['# Interactions']
    ax.legend(lines, labels,
              loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, frameon=False)

    # Add subplot title
    ax.text(0.5, 1.1, title,
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontweight='bold')


# Example usage with empty data - this function can be called with real data
# plot_dataset(ax1, [], [], {}, "(a) Bitcoin-OTC")
# plot_dataset(ax2, [], [], {}, "(b) Bitcoin-Alpha")

# Save image if needed with real data
# plt.savefig('attention_interaction_plot.png',
#            bbox_inches='tight',
#            pad_inches=0.2,
#            transparent=False)
# plt.show()