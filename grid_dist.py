import json
import seaborn as sns
from diffusion import Grid
from matplotlib import pyplot as plt

def plot_grid_distribution(grid, bins=50):
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

   colors = ['red', 'green', 'blue']

   for i in range(grid.feature_dim):
       sns.histplot(grid.grid[0,i].flatten(), bins=bins, ax=ax1,
                   color=colors[i], alpha=0.5, label=f'Channel {i}')
       sns.kdeplot(grid.grid[0,i].flatten(), ax=ax2,
                   color=colors[i], label=f'Channel {i}')

   ax1.set_title('Value Distribution per Channel')
   ax2.set_title('Density Distribution per Channel')
   ax1.legend()
   ax2.legend()

   plt.tight_layout()
   plt.show()

load_file = "simulation_20241217_012214.json"
with open(load_file, 'r') as f:
    saved_data = json.load(f)
    grid = Grid(saved_state = saved_data["grid"])

def count_exceeding_values(grid, threshold=128):
    counts = []
    for i in range(grid.feature_dim):
        count = (grid.grid[0,i] > threshold).sum()
        print(f"Channel {i}: {count} cells exceed {threshold}")
        counts.append(count)
    return counts

if __name__ == '__main__':
    print("Red total:", grid.grid[0][0].sum()/(192 * 108))
    print("Green total:", grid.grid[0][1].sum()/(192 * 108))
    print("Blue total:", grid.grid[0][2].sum()/(192 * 108))
    # count_exceeding_values(grid)
    plot_grid_distribution(grid)
