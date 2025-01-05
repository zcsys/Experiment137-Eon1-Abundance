from base_vars import *
from base_vars import METABOLIC_ACTIVITY, AUTO_FISSION_THRESHOLD
from helpers import *
import torch

def Rules(simul, n):
    global METABOLIC_ACTIVITY, AUTO_FISSION_THRESHOLD

    # Coming into existence and perishing
    if 0 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if mask:
                simul.things.monad_division(i)
        if simul.period > 0 or simul.epoch > 0:
            simul.things.energies -= METABOLIC_ACTIVITY
        else:
            simul.things.energies[40:] -= METABOLIC_ACTIVITY
        to_remove = torch.nonzero(simul.things.energies <= 0)
        if len(to_remove) > 0:
            simul.things.perish_monad(to_remove.squeeze(1).tolist())
        simul.things.E = simul.things.energies.sum().item() // 1000

    # Population control
    if 1 in n:
        if simul.period > 0 or simul.epoch > 0:
            if simul.things.E <= 100:
                METABOLIC_ACTIVITY = 0.1
            elif 100 < simul.things.E <= 200:
                METABOLIC_ACTIVITY = 0.1 + 0.009 * (simul.things.E - 100)
            elif 200 < simul.things.E:
                METABOLIC_ACTIVITY = 1. + 0.09 * (simul.things.E - 200)
        else:
            if simul.things.E <= 400:
                METABOLIC_ACTIVITY = 0.1
            elif 400 < simul.things.E <= 500:
                METABOLIC_ACTIVITY = 0.1 + 0.009 * (simul.things.E - 400)
            elif 500 < simul.things.E:
                METABOLIC_ACTIVITY = 1. + 0.09 * (simul.things.E - 500)

    # Resource management
    if 2 in n:
        simul.things.add_energyUnits_atGridCells(simul.grid.grid[0][1],
                                                 ENERGY_THRESHOLD)

    # Bond formation and breaking
    if 3 in n:
        # Get structural unit indices and distances
        struct_mask = simul.things.structure_mask
        if struct_mask.any():
            # Recompute distances for current configuration
            _, distances, _ = vicinity(simul.things.positions)
            str_indices = torch.nonzero(struct_mask).squeeze(1)
            struct_distances = distances[struct_mask][:, struct_mask]

            # Create probability matrices for bond formation and breaking
            form_prob_matrix = torch.rand_like(struct_distances)
            break_prob_matrix = torch.rand_like(struct_distances)

            # Find pairs for bond formation that meet criteria
            valid_pairs = torch.nonzero(
                (struct_distances >= 10) &  # min_dist
                (struct_distances <= 50) &  # max_dist
                (form_prob_matrix < 0.0001) &  # probability threshold
                (struct_distances > 0)  # exclude self-bonds
            )

            # Attempt to form bonds for valid pairs
            for i, j in valid_pairs:
                simul.things.bonds.form(i, j,
                                        simul.things.positions[struct_mask])

            # Check existing bonds for breaking
            bonds = simul.things.bonds.bonds
            for i in range(len(bonds)):
                for j, bonded_idx in enumerate(bonds[i]):
                    if bonded_idx == torch.inf:
                        continue

                    bonded_idx = bonded_idx.long()
                    dist = struct_distances[i, bonded_idx]

                    # Breaking probability increases as distance approaches max
                    if dist > 40:
                        break_prob = 0.0001 * (dist - 40)
                        if break_prob_matrix[i, j] < break_prob:
                            simul.things.bonds.break_bond(i, bonded_idx)
