from base_vars import *
from base_vars import METABOLIC_ACTIVITY, AUTO_FISSION_THRESHOLD
from helpers import *
import torch

def Rules(simul, n):
    global METABOLIC_ACTIVITY, AUTO_FISSION_THRESHOLD

    # Coming into existence and perishing
    if 0 in n:
        change_occured = False
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        if fission_mask.any():
            for i in fission_mask.nonzero().squeeze(1):
                simul.things.monad_division(i)
            change_occured = True
        if simul.period > 0 or simul.epoch > 0:
            simul.things.energies -= METABOLIC_ACTIVITY
        else:
            simul.things.energies[40:] -= METABOLIC_ACTIVITY
        to_remove = torch.nonzero(simul.things.energies <= 0)
        if len(to_remove) > 0:
            simul.things.perish_monad(to_remove.squeeze(1).tolist())
            change_occured = True
        simul.things.E = simul.things.energies.sum().item() // 1000
        if change_occured:
            _, simul.things.distances, simul.things.diffs = vicinity(
                simul.things.positions
            )

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
        if simul.things.add_energyUnits_atGridCells(simul.grid.grid[0][1],
                                                    ENERGY_THRESHOLD):
            _, simul.things.distances, simul.things.diffs = vicinity(
                simul.things.positions
            )

    # Bond formation and breaking
    if 3 in n:
        # Get structural unit indices and distances
        struct_mask = simul.things.structure_mask
        if struct_mask.any():
            str_indices = torch.nonzero(struct_mask).squeeze(1)
            str_distances = simul.things.distances[struct_mask][:, struct_mask]

            # Create probability matrices for bond formation and breaking
            form_prob_matrix = torch.rand_like(str_distances)
            break_prob_matrix = torch.rand_like(str_distances)

            # Find pairs for bond formation that meet criteria
            valid_pairs = torch.nonzero(
                (str_distances >= 10) & (str_distances <= 50) &
                (form_prob_matrix < 0.01)
            )

            # Attempt to form bonds for valid pairs
            for i, j in valid_pairs:
                simul.things.bonds.form_str_bond(
                    i, j, simul.things.positions[struct_mask]
                )

            # Check existing bonds for breaking
            str_bonds = simul.things.bonds.bonds[:, :2]
            for i in range(len(str_bonds)):
                for j, bonded_idx in enumerate(str_bonds[i]):
                    if bonded_idx == torch.inf:
                        continue

                    bonded_idx = bonded_idx.long()
                    dist = str_distances[i, bonded_idx]

                    # Breaking probability increases as distance approaches max
                    if dist > 50:
                        break_prob = 0.02 * (dist - 50)
                        if break_prob_matrix[i, j] < break_prob:
                            simul.things.bonds.break_str_bond(i, bonded_idx)

            # Handle monad bonds if there are monads
            if simul.things.monad_mask.any():
                str_positions = simul.things.positions[struct_mask]
                mnd_positions = simul.things.positions[simul.things.monad_mask]
                mnd_str_distances = simul.things.distances[
                    simul.things.monad_mask
                ][:, struct_mask]

                # Form new monad bonds
                active_bond_sites = simul.things.bond_sites > 0
                for mnd_idx in active_bond_sites.nonzero():
                    mnd_idx = mnd_idx.item()

                    # Find eligible structural units within range
                    valid_dists = mnd_str_distances[mnd_idx]
                    valid_mask = (valid_dists >= 10) & (valid_dists <= 50)

                    # Only try to bond if there are valid candidates
                    if valid_mask.any():
                        # Calculate bonding probabilities
                        eligible_mask = (
                            torch.rand(len(valid_mask)) <
                            simul.things.bond_sites[mnd_idx]
                        ) & valid_mask

                        if eligible_mask.any():
                            # Try to bond with closest eligible unit
                            valid_dists[~eligible_mask] = torch.inf
                            closest_str_idx = valid_dists.argmin().item()

                            simul.things.bonds.form_mnd_bond(
                                closest_str_idx,
                                simul.things.universal_monad_identifier[
                                    mnd_idx
                                ],
                                str_positions,
                                mnd_positions[mnd_idx]
                            )

                # Check existing monad bonds for breaking
                for str_idx in range(len(str_positions)):
                    mnd_bond = simul.things.bonds.bonds[str_idx, 2]
                    if mnd_bond != torch.inf:
                        mnd_idx = torch.where(
                            simul.things.universal_monad_identifier == mnd_bond
                        )[0][0].long()
                        dist = torch.norm(str_positions[str_idx] -
                                          mnd_positions[mnd_idx])

                        if (dist > 50 and
                            torch.rand(1) < 0.02 * (dist - 50) or
                            dist < 30 and
                            simul.things.bond_sites[mnd_idx] < 0):
                            simul.things.bonds.break_mnd_bond(str_idx, 0)
