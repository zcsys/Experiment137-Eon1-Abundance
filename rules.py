from base_vars import *
from base_vars import METABOLIC_ACTIVITY, AUTO_FISSION_THRESHOLD

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
            simul.things.energies[50:] -= METABOLIC_ACTIVITY
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
            if simul.things.E <= 500:
                METABOLIC_ACTIVITY = 0.1
            elif 500 < simul.things.E <= 600:
                METABOLIC_ACTIVITY = 0.1 + 0.009 * (simul.things.E - 500)
            elif 600 < simul.things.E:
                METABOLIC_ACTIVITY = 1. + 0.09 * (simul.things.E - 600)

    # Resource management
    if 2 in n:
        simul.things.add_energyUnits_atGridCells(simul.grid.grid[0][1],
                                                 ENERGY_THRESHOLD)
