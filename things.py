import torch
import pygame
import random
import math
import json
from base_vars import *
from helpers import *
from nn import *
from simulation import draw_dashed_circle
from diffusion import Grid

class Bonds:
    def __init__(self, StrPop, valence = 3, min_angle = 120, max_dist = 50):
        self.StrPop = StrPop
        self.valence = valence
        self.min_angle = min_angle * math.pi / 180
        self.max_dist = max_dist
        self.bonds = torch.tensor(torch.inf).repeat(self.StrPop, self.valence)

    def get_first_available_slot_str(self, i):
        available_slots = torch.where(self.bonds[i][:2] == torch.inf)[0]
        return available_slots[0].item() if len(available_slots) > 0 else None

    def get_first_available_slot_mnd(self, i):
        available_slots = torch.where(self.bonds[i][2:] == torch.inf)[0]
        return available_slots[0].item() if len(available_slots) > 0 else None

    def form_str_bond(self, i, j, positions):
        if i == j or (self.bonds[i] == j).any() or (self.bonds[j] == i).any():
            return

        i_slot = self.get_first_available_slot_str(i)
        j_slot = self.get_first_available_slot_str(j)

        if (i_slot is not None and j_slot is not None and
            self.check_constraints_for_str_bond(i, j, positions)):
            self.bonds[i][i_slot] = j
            self.bonds[j][j_slot] = i

    def form_mnd_bond(self, i, j, str_positions, mnd_position):
        if (self.bonds[i][2:] == j).any():
            return
        slot = self.get_first_available_slot_mnd(i)
        if (slot is not None and
            self.check_constraints_for_mnd_bond(i, str_positions,
                                                mnd_position)):
            self.bonds[i][slot + 2] = j

    def break_str_bond(self, i, j):
        i_slot = torch.where(self.bonds[i] == j)[0]
        j_slot = torch.where(self.bonds[j] == i)[0]
        if len(i_slot) > 0:
            self.bonds[i][i_slot] = torch.inf
            self.bonds[j][j_slot] = torch.inf
            self.bonds[i, 2:] = torch.inf
            self.bonds[j, 2:] = torch.inf

    def break_mnd_bond(self, i, j):
        self.bonds[i, j + 2] = torch.inf

    def check_constraints_for_str_bond(self, i, j, pos):
        if torch.norm(pos[j] - pos[i]) > self.max_dist:
            return False

        i_bonded = self.bonds[i][self.bonds[i] != torch.inf]
        if (len(i_bonded) > 0 and
            angle(pos[i], pos[i_bonded[0].long()], pos[j]) < self.min_angle):
            return False

        j_bonded = self.bonds[j][self.bonds[j] != torch.inf]
        if (len(j_bonded) > 0 and
            angle(pos[j], pos[j_bonded[0].long()], pos[i]) < self.min_angle):
            return False

        return True

    def check_constraints_for_mnd_bond(self, i, str_positions, mnd_position):
        if ((self.bonds[i][:2] == torch.inf).any() or
            torch.norm(str_positions[i] - mnd_position) > self.max_dist):
            return False

        return True

class Things:
    def __init__(self, thing_types = None, state_file = None):
        # Initialize font
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)

        if state_file:
            self.load_state(state_file)
            return

        # Place memory units
        # self.place_memoryUnits()

        # Main attributes
        self.thing_types = thing_types
        self.sizes = torch.tensor([THING_TYPES[x]["size"] for x in thing_types])
        self.positions = add_positions(len(thing_types))
        self.colors = [THING_TYPES[x]["color"] for x in thing_types]

        # Initialize tensor masks
        self.monad_mask = torch.tensor(
            [thing_type == "monad" for thing_type in self.thing_types]
        )
        self.energy_mask = torch.tensor(
            [thing_type == "energyUnit" for thing_type in self.thing_types]
        )
        self.structure_mask = torch.tensor(
            [thing_type == "structuralUnit" for thing_type in self.thing_types]
        )
        self.memory_mask = torch.tensor(
            [thing_type == "memoryUnit" for thing_type in self.thing_types]
        )

        # Initialize state vars
        self.N = len(self.thing_types)
        self.Pop = self.monad_mask.sum().item()
        self.energies = torch.tensor(
            [THING_TYPES["monad"]["initial_energy"]
            for _ in range(self.Pop)]
        )
        self.E = self.energies.sum().item() // 1000
        self.memory = torch.zeros((self.Pop, 12), dtype = torch.float32)
        self.str_manipulations = torch.zeros((0, 2, 3), dtype = torch.float32)
        self.Rotation = torch.rand((self.Pop,)) * 2 * math.pi
        self.U = torch.stack(
            (
                torch.cos(self.Rotation),
                torch.sin(self.Rotation)
            ),
            dim = 1
        )
        self.distances = None
        self.bonds = torch.empty(0)
        self.bond_sites = torch.zeros((self.Pop,), dtype = torch.float32)
        self.universal_monad_identifier = torch.arange(self.Pop)
        self.total_number_of_all_monads = self.Pop

        # Initialize genomes and lineages
        self.genomes = torch.cat(
            (
                torch.zeros((self.Pop, 12), dtype = torch.float32),
                initialize_parameters(self.Pop, 52, 46, "nn23")
            ),
            dim = 1
        )
        print("Genome size:", self.genomes.shape[1])
        self.lineages = [[0] for _ in range(self.Pop)]
        self.apply_genomes()

    def from_general_to_monad_idx(self, i):
        return self.monad_mask[:i].sum().item()

    def from_monad_to_general_idx(self, i):
        return torch.nonzero(self.monad_mask)[i].item()

    def get_generation(self, i):
        return self.lineages[i][0] + len(self.lineages[i])

    def apply_genomes(self):
        """Monad1XC814 neurogenetics"""
        self.elemental_biases = torch.tanh(self.genomes[:, :12])
        self.nn = nn23(self.genomes[:, 12:], 52, 46)

    def mutate(self, i, probability = 0.1, strength = 1.):
        mutation_mask = torch.rand_like(self.genomes[i]) < probability
        mutations = torch.rand_like(self.genomes[i]) * 2 - 1
        return self.genomes[i] + mutation_mask * mutations * strength

    def sensory_inputs(self, grid):
        # For each monad, there's a vector pointing towards the center of the
        # universe, with increasing magnitude as the thing gets closer to edges.
        if self.monad_mask.any():
            midpoint = torch.tensor([SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2])
            col0 = (1 - self.positions[self.monad_mask] / midpoint)
        else:
            col0 = torch.zeros((self.Pop, 2))

        # For each monad, the combined effect of energy particles in their
        # vicinity is calculated.
        if self.distances == None:
            _, self.distances, self.diffs = vicinity(self.positions)
        else:
            pass
        if self.monad_mask.any() and self.energy_mask.any():
            col1  = (
                self.diffs[self.monad_mask][:, self.energy_mask] /
                (
                    self.distances[self.monad_mask][
                        :, self.energy_mask
                    ] ** 2 + epsilon
                ).unsqueeze(2)
            ).sum(dim = 1) * STD_RADIUS
        else:
            col1 = torch.zeros((self.Pop, 2))

        # For each monad, the combined effect of other monads in their vicinity
        # is calculated.
        if self.Pop > 1:
            col2  = (
                self.diffs[self.monad_mask][:, self.monad_mask] /
                (
                    self.distances[self.monad_mask][
                        :, self.monad_mask
                    ] ** 2 + epsilon
                ).unsqueeze(2)
            ).sum(dim = 1) * STD_RADIUS
        else:
            col2 = torch.zeros((self.Pop, 2))

        # Resource & gradient sensors
        y_pos = (self.positions[self.monad_mask][:, 1] // grid.cell_size).int()
        x_pos = (self.positions[self.monad_mask][:, 0] // grid.cell_size).int()
        col3 = grid.grid[0, :, y_pos, x_pos].T / 255

        grad_x, grad_y = grid.gradient()
        col4 = torch.stack(
            (
                grad_x[0, 0, y_pos, x_pos],
                grad_y[0, 0, y_pos, x_pos],
                grad_x[0, 1, y_pos, x_pos],
                grad_y[0, 1, y_pos, x_pos],
                grad_x[0, 2, y_pos, x_pos],
                grad_y[0, 2, y_pos, x_pos]
            ),
            dim = 1
        ) / 255

        # Monads can interact with at most 6 structural units
        if self.monad_mask.any() and self.structure_mask.any():
            dist = self.distances[self.monad_mask][:, self.structure_mask]
            self.dist_mnd_str, self.structure_indices = torch.topk(
                dist.masked_fill(dist == 0, torch.inf),
                k = min(6, self.structure_mask.sum()),
                dim = 1,
                largest = False
            )

            col5  = (
                torch.gather(
                    self.diffs[self.monad_mask],
                    1,
                    self.structure_indices.unsqueeze(2).expand(-1, -1, 2)
                ) / (
                    torch.gather(
                        self.distances[self.monad_mask],
                        1,
                        self.structure_indices
                    ) ** 2 + epsilon
                ).unsqueeze(2)
            ).view(self.Pop, 12) * STD_RADIUS
        else:
            col5 = torch.zeros((self.Pop, 12), dtype = torch.float32)

        # Combine the inputs to create the final input tensor
        self.input_vectors = torch.cat(
            (
                self.elemental_biases,
                decompose_vectors(
                    torch.cat((col0, col1, col2, col4, col5), dim = 1),
                    self.U
                ),
                col3,
                (self.energies / 10000).unsqueeze(1),
                self.memory
            ),
            dim = 1
        ).view(self.Pop, 52, 1)

    def neural_action(self):
        return self.nn.forward(self.input_vectors)

    def random_action(self):
        numberOf_energyUnits = self.energy_mask.sum().item()
        if SYSTEM_HEAT == 0:
            return torch.tensor([[0, 0] for _ in range(numberOf_energyUnits)],
                                dtype = torch.float32)
        values = (
            torch.tensor(
                list(
                    range(SYSTEM_HEAT * 2 + 1)
                ),
                dtype = torch.float32
            ) - SYSTEM_HEAT
        )
        weights = torch.ones(SYSTEM_HEAT * 2 + 1, dtype = torch.float32)
        indices = torch.multinomial(
            weights,
            numberOf_energyUnits * 2,
            replacement = True
        ).view(numberOf_energyUnits, 2)
        return values[indices]

    def re_action(self, grid, neural_action):
        # Helper variables
        numberOf_structuralUnits = self.structure_mask.sum()
        expanded_indices = self.structure_indices.unsqueeze(2).expand(-1, -1, 2)

        # Initialize force field
        force_field = torch.zeros_like(
            grid.grid
        ).repeat(2, 1, 1, 1)
        indices = (self.positions[self.structure_mask] // grid.cell_size).long()

        numerator = torch.gather(
            self.diffs[self.monad_mask][:, self.structure_mask],
            1,
            expanded_indices
        )

        denominator = (
            torch.gather(
                self.distances[self.monad_mask][:, self.structure_mask],
                1,
                self.structure_indices
            ) + epsilon
        ).unsqueeze(2)

        unit_vectors = numerator / denominator

        perpendicular = torch.stack(
            (
                -unit_vectors[..., 1],
                unit_vectors[..., 0]
            ),
            dim = 2
        )

        # Calculate resource manipulations
        manipulation_contributions = (
            neural_action[:, 12:30].view(-1, 6, 1, 3) *
            (
                unit_vectors /
                denominator
            ).unsqueeze(3)
        ) * STD_RADIUS

        self.str_manipulations.scatter_add_(
            0,
            expanded_indices.repeat(1, 1, 3).view(-1, 2, 3),
            manipulation_contributions.view(-1, 2, 3)
        ).clamp_(-10, 10)

        # Calculate and apply force field with diffusion
        for i in range(2): # For vertical and horizontal axes
            for j in range(3): # For each channel
                force_field[i, j][
                    indices[:, 1], indices[:, 0]
                ] += self.str_manipulations[:, i, j]

        grid.diffuse(force_field)

        # Calculate movements
        movement_contributions = (
            (
                neural_action[:, 0:6].unsqueeze(2) *
                unit_vectors +
                neural_action[:, 6:12].unsqueeze(2) *
                perpendicular
            ) / denominator
        ) * STD_RADIUS

        # Reduce energies
        self.energies -= (
            movement_contributions.norm(dim = 2)
        ).sum(dim = 1) / 6

        # Initialize movement tensor with repulsive force
        movement_tensor = (
            -self.diffs[self.structure_mask][:, self.structure_mask] /
            (
                self.distances[self.structure_mask][
                    :, self.structure_mask
                ] ** 2 + epsilon
            ).unsqueeze(2)
        ).sum(dim = 1)

        # Return movements
        return movement_tensor.scatter_add(
            0,
            expanded_indices.view(-1, 2),
            movement_contributions.view(-1, 2)
        ) * SYSTEM_HEAT

    def rotation_and_movement(self, neural_action):
        self.Rotation += torch.where(
            neural_action[:, 1] <= 0,
            neural_action[:, 0],
            torch.tensor([0.])
        )
        self.U = torch.stack(
            (
                torch.cos(self.Rotation),
                torch.sin(self.Rotation)
            ),
            dim = 1
        )
        movements = torch.clamp(
            neural_action[:, 1],
            min = 0
        ).unsqueeze(1) * self.U * 5.
        self.energies -= torch.norm(movements, dim = 1)
        return movements

    def background_repulsion(self, radius = STD_RADIUS):
        self.moving_mask = self.monad_mask | self.structure_mask
        self.movement_tensor[self.moving_mask] -= (
            self.diffs[self.moving_mask][:, self.moving_mask] /
            (
                self.distances[self.moving_mask][:, self.moving_mask] + epsilon
            ).unsqueeze(2) *
            (
                radius - self.distances[self.moving_mask][:, self.moving_mask]
            ).clamp(0, radius).unsqueeze(2)
        ).sum(dim = 1)
        self.movement_tensor[self.energy_mask] -= (
            self.diffs[self.energy_mask][:, self.structure_mask] /
            (
                self.distances[self.energy_mask][:, self.structure_mask] +
                epsilon
            ).unsqueeze(2) *
            (
                radius -
                self.distances[self.energy_mask][:, self.structure_mask]
            ).clamp(0, radius).unsqueeze(2)
        ).sum(dim = 1)

    def bond_adjustment(self):
        # Get indices
        str_bonds = self.bonds.bonds[:, :2]
        valid_bonds = str_bonds != torch.inf
        bond_pairs = valid_bonds.nonzero()
        if len(bond_pairs) == 0:
            return
        bonded_idx = bond_pairs[:, 0]
        bonders_idx = str_bonds[valid_bonds].long()
        pos = self.positions[self.structure_mask]

        # Get positions
        start_pos = pos[bonded_idx]
        end_pos = pos[bonders_idx]
        bond_centers = (start_pos + end_pos) / 2
        current_lengths = torch.norm(end_pos - start_pos, dim = 1)
        half_lengths = current_lengths / 2

        # Bond repulsion
        moving_positions = self.positions[self.moving_mask]
        _, distances, diffs = vicinity(moving_positions, radius = 30,
                                       target_positions = bond_centers)
        expanded_half_lengths = half_lengths.unsqueeze(0).expand(
            len(moving_positions), -1
        )
        self.movement_tensor[self.moving_mask] -= (
            diffs / (distances + epsilon).unsqueeze(2) *
            (
                expanded_half_lengths - distances
            ).clamp(torch.tensor(0.), expanded_half_lengths).unsqueeze(2)
        ).sum(dim = 1)

        # Distance adjustment
        target_radius = 15
        direction_vectors = (end_pos - start_pos) / \
                            (current_lengths.unsqueeze(1) + epsilon)
        apply_mask = (half_lengths > target_radius).unsqueeze(1)
        full_indices = self.structure_mask.nonzero().expand(-1, 2)
        self.movement_tensor.scatter_add_(
            0,
            full_indices[bonded_idx],
            torch.where(apply_mask, direction_vectors, -direction_vectors)
        )

        # Angle adjustment
        has_two_bonds = (valid_bonds.sum(dim = 1) == 2)
        if has_two_bonds.any():
            unit_str_idx = has_two_bonds.nonzero().squeeze(1)
            unit_pos = pos[has_two_bonds]
            vec1 = pos[str_bonds[has_two_bonds][:, 0].long()] - unit_pos
            vec2 = pos[str_bonds[has_two_bonds][:, 1].long()] - unit_pos
            vec1 /= torch.norm(vec1, dim = 1, keepdim = True)
            vec2 /= torch.norm(vec2, dim = 1, keepdim = True)
            angles = torch.acos(torch.sum(vec1 * vec2, dim = 1))
            angles_to_adjust = angles < self.bonds.min_angle

            bisectors = (vec1 + vec2)[angles_to_adjust]
            bisectors /= torch.norm(bisectors, dim = 1, keepdim = True) + \
                         epsilon
            self.movement_tensor.scatter_add_(
                0,
                full_indices[unit_str_idx][angles_to_adjust],
                bisectors * 5.
            )

        # Handle monad bonds
        mnd_bonds = self.bonds.bonds[:, 2:]
        valid_bonds = mnd_bonds != torch.inf
        bond_pairs = valid_bonds.nonzero()
        if len(bond_pairs) == 0:
            return
        bonded_idx = bond_pairs[:, 0]
        bonders_idx = mnd_bonds[valid_bonds].long()
        bonders_idx = torch.where(
            (
                self.universal_monad_identifier.unsqueeze(0) ==
                mnd_bonds[valid_bonds].unsqueeze(1)
            )
        )[1].long()
        start_pos = pos[bonded_idx]
        end_pos = self.positions[self.monad_mask][bonders_idx]
        bond_centers = (start_pos + end_pos) / 2
        current_lengths = torch.norm(end_pos - start_pos, dim = 1)
        half_lengths = current_lengths / 2
        adjustment_vectors = (end_pos - start_pos) / \
                             (current_lengths.unsqueeze(1) + epsilon)
        apply_coeff = (target_radius - half_lengths).unsqueeze(1)
        full_indices = self.monad_mask.nonzero().expand(-1, 2)
        self.movement_tensor.scatter_add_(
            0,
            full_indices[bonders_idx],
            apply_coeff * adjustment_vectors
        )

    def final_action(self, grid):
        # Update sensory inputs
        self.sensory_inputs(grid)

        # Initialize the movement tensor for this step
        if self.N > 0:
            self.movement_tensor = torch.zeros(
                (self.N, 2),
                dtype = torch.float32
            )

        # Monad movements & memory
        if self.monad_mask.any():
            neural_action = self.neural_action()
            self.movement_tensor[self.monad_mask] = self.rotation_and_movement(
                neural_action[:, :2]
            )
            self.memory = neural_action[:, 33:45]
            self.bond_sites = neural_action[:, 45]

        # Fetch energyUnit movements
        if self.energy_mask.any():
            self.movement_tensor[self.energy_mask] = self.random_action()

        # Fetch structuralUnit reactions
        if self.structure_mask.any():
            if self.Pop > 0:
                self.movement_tensor[self.structure_mask] = self.re_action(
                    grid,
                    neural_action[:, 3:33]
                ).clamp_(-SYSTEM_HEAT, SYSTEM_HEAT)
            else:
                self.movement_tensor[self.structure_mask] = torch.zeros(
                    (self.structure_mask.sum(), 2),
                    dtype = torch.float32
                )

        # Self-induced divison
        if self.monad_mask.any():
            fission = neural_action[:, 2] > torch.rand(self.Pop)
            for i in fission.nonzero():
                self.monad_division(i.item())
            if fission.any():
                _, self.distances, self.diffs = vicinity(self.positions)

        # Calculate final forces and apply movements
        self.background_repulsion()
        self.bond_adjustment()
        self.update_positions()

        # Update total monad energy
        self.E = self.energies.sum().item() // 1000

    def update_positions(self):
        # Apply movement tensor
        self.positions = self.positions + self.movement_tensor
        self.positions = torch.stack(
            [
                torch.clamp(
                    self.positions[:, 0],
                    min = self.sizes,
                    max = SIMUL_WIDTH - self.sizes
                ),
                torch.clamp(
                    self.positions[:, 1],
                    min = self.sizes,
                    max = SIMUL_HEIGHT - self.sizes
                )
            ],
            dim = 1
        )
        _, self.distances, self.diffs = vicinity(self.positions)

        # EnergyUnit-monad collisions
        energy_monad_dist = self.distances[self.energy_mask][:, self.monad_mask]
        energy_collision_mask = (
            (0. < energy_monad_dist) &
            (energy_monad_dist < (THING_TYPES["monad"]["size"]))
        )

        if energy_collision_mask.any():
            energy_idx, monad_idx = energy_collision_mask.nonzero(
                as_tuple = True
            )
            self.energies.scatter_add_(
                0,
                monad_idx,
                torch.tensor(UNIT_ENERGY).expand_as(monad_idx)
            )
            energy_idx_general = torch.where(self.energy_mask)[0][energy_idx]
            self.remove_energyUnits(unique(energy_idx_general.tolist()))
            _, self.distances, self.diffs = vicinity(self.positions)

    def monad_division(self, i):
        # Set out main attributes and see if division is possible
        thing_type = "monad"
        initial_energy = self.energies[i] / 2
        if (initial_energy <
            torch.tensor(THING_TYPES[thing_type]["initial_energy"])):
            return 0
        size = THING_TYPES[thing_type]["size"]
        idx = self.from_monad_to_general_idx(i)
        x, y = tuple(self.positions[idx].tolist())
        angle = random.random() * 2 * math.pi
        new_position = torch.tensor([
            x + math.cos(angle) * (size + 1) * 2,
            y + math.sin(angle) * (size + 1) * 2
        ])
        dist_mnd = torch.norm(
            self.positions[self.monad_mask] - new_position, dim = 1
        )
        dist_str = torch.norm(
            self.positions[self.structure_mask] - new_position, dim = 1
        )
        if (new_position[0] < size or new_position[0] > SIMUL_WIDTH - size or
            new_position[1] < size or new_position[1] > SIMUL_HEIGHT - size or
            (dist_str < size + THING_TYPES["structuralUnit"]["size"]).any() or
            (dist_mnd < size * 2).any()):
            return 0

        # Create a new set of attributes
        self.thing_types.append(thing_type)
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(size).unsqueeze(0)
            ),
            dim = 0
        )
        self.positions = torch.cat(
            (
                self.positions,
                new_position.unsqueeze(0)
            ),
            dim = 0
        )
        self.energies[i] -= initial_energy
        self.energies = torch.cat(
            (
                self.energies,
                initial_energy.unsqueeze(0)
            ),
            dim = 0
        )
        self.memory = torch.cat(
            (
                self.memory,
                torch.zeros((1, 12), dtype = torch.float32)
            ),
            dim = 0
        )
        rotation = torch.rand((1,)) * 2 * math.pi
        self.Rotation = torch.cat(
            (
                self.Rotation,
                rotation
            ),
            dim = 0
        )
        self.U = torch.cat(
            (
                self.U,
                torch.stack(
                    (
                        torch.cos(rotation),
                        torch.sin(rotation)
                    ),
                    dim = 1
                )
            ),
            dim = 0
        )
        self.bond_sites = torch.cat(
            (
                self.bond_sites,
                torch.tensor([0.])
            ),
            dim = 0
        )
        self.universal_monad_identifier = torch.cat(
            (
                self.universal_monad_identifier,
                torch.tensor([self.total_number_of_all_monads])
            ),
            dim = 0
        )
        self.movement_tensor = torch.cat(
            (
                self.movement_tensor,
                torch.tensor([[0., 0.]])
            ),
            dim = 0
        )
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.tensor([True])
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.tensor([False])
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.tensor([False])
            ),
            dim = 0
        )
        self.memory_mask = torch.cat(
            (
                self.memory_mask,
                torch.tensor([False])
            ),
            dim = 0
        )
        self.N += 1
        self.Pop += 1
        self.total_number_of_all_monads += 1

        # Mutate the old genome & apply the new genome
        idx = self.from_general_to_monad_idx(i)
        genome = self.mutate(idx)
        self.genomes = torch.cat(
            (
                self.genomes,
                genome.unsqueeze(0)
            ),
            dim = 0
        )
        self.apply_genomes()
        if genome is self.genomes[idx]:
            self.lineages.append(self.lineages[idx])
            self.colors.append(self.color[i])
        else:
            new_lineage = self.lineages[idx] + [0]
            while True:
                new_lineage[-1] += 1
                if new_lineage not in self.lineages:
                    break
            self.lineages.append(new_lineage)
            self.colors.append(get_color_by_genome(genome))
            # print(new_lineage)

        return 1

    def perish_monad(self, indices):
        for i in indices[::-1]:
            # Remove monad-only attributes
            self.genomes = remove_element(self.genomes, i)
            self.energies = remove_element(self.energies, i)
            self.memory = remove_element(self.memory, i)
            self.Rotation = remove_element(self.Rotation, i)
            self.U = remove_element(self.U, i)
            self.bond_sites = remove_element(self.bond_sites, i)
            del self.lineages[i]

            # Update bonds
            universalID = self.universal_monad_identifier[i]
            self.universal_monad_identifier = remove_element(
                self.universal_monad_identifier, i
            )
            self.bonds.bonds[:, 2:] = torch.where(
                self.bonds.bonds[:, 2:] == universalID,
                torch.inf,
                self.bonds.bonds[:, 2:]
            )

            # Get general index to remove universal attributes
            idx = self.from_monad_to_general_idx(i)

            # Update main attributes and state vars
            del self.thing_types[idx]
            del self.colors[idx]
            self.sizes = remove_element(self.sizes, idx)
            self.positions = remove_element(self.positions, idx)
            self.monad_mask = remove_element(self.monad_mask, idx)
            self.energy_mask = remove_element(self.energy_mask, idx)
            self.structure_mask = remove_element(self.structure_mask, idx)
            self.memory_mask = remove_element(self.memory_mask, idx)

        # Update collective state vars
        self.N -= len(indices)
        self.Pop -= len(indices)

        self.apply_genomes()

    def add_energyUnits(self, N):
        for _ in range(N):
            self.thing_types.append("energyUnit")
            self.colors.append(THING_TYPES["energyUnit"]["color"])
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(
                    [THING_TYPES["energyUnit"]["size"] for _ in range(N)]
                )
            ),
            dim = 0
        )
        self.positions = add_positions(N, self.positions)
        self.N += N
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.ones(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.memory_mask = torch.cat(
            (
                self.memory_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )

    def add_energyUnits_atGridCells(self, feature, threshold):
        cell_indices = (feature >= threshold).nonzero()
        occupied_grid_cells = self.positions // GRID_CELL_SIZE

        positions_to_add = torch.empty((0, 2), dtype = torch.float32)
        for y, x in cell_indices:
            new_pos = torch.tensor([x, y])
            if torch.any(
                torch.all(occupied_grid_cells.int() == new_pos, dim = 1)
            ):
                continue
            else:
                positions_to_add = torch.cat(
                    (
                        positions_to_add,
                        (GRID_CELL_SIZE * (new_pos + 0.5)).unsqueeze(0)
                    ),
                    dim = 0
                )
                feature[y, x] = RESOURCE_TARGET

        N = len(positions_to_add)
        if N == 0:
            return 0
        self.N += N

        for _ in range(N):
            self.thing_types.append("energyUnit")
            self.colors.append(THING_TYPES["energyUnit"]["color"])
        self.positions = torch.cat(
            (
                self.positions,
                positions_to_add
            ),
            dim = 0
        )
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(
                    THING_TYPES["energyUnit"]["size"]
                ).expand(N)
            ),
            dim = 0
        )
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.ones(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.memory_mask = torch.cat(
            (
                self.memory_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )

        return 1

    def remove_energyUnits(self, indices):
        for i in indices[::-1]:
            del self.thing_types[i]
            del self.colors[i]

        mask = torch.ones(self.N, dtype = torch.bool)
        mask[indices] = False
        self.N = mask.sum().item()

        self.sizes = self.sizes[mask]
        self.positions = self.positions[mask]
        self.monad_mask = self.monad_mask[mask]
        self.energy_mask = self.energy_mask[mask]
        self.structure_mask = self.structure_mask[mask]
        self.memory_mask = self.memory_mask[mask]

    def draw(self, screen, show_info = True, show_sight = False):
        # Draw bonds
        if self.structure_mask.any():
            struct_positions = self.positions[self.structure_mask]
            struct_indices = torch.where(self.structure_mask)[0]

            if hasattr(self.bonds, 'bonds'):
                for i in range(len(self.bonds.bonds)):
                    for j, bonded_idx in enumerate(self.bonds.bonds[i, :2]):
                        if bonded_idx == torch.inf:
                            continue
                        if i < bonded_idx:
                            start_pos = struct_positions[i]
                            end_pos = struct_positions[bonded_idx.long()]
                            pygame.draw.line(
                                screen,
                                colors["GB"],
                                (
                                    int(start_pos[0].item()),
                                    int(start_pos[1].item())
                                ),
                                (
                                    int(end_pos[0].item()),
                                    int(end_pos[1].item())
                                ),
                                1
                            )

                    for j, bonded_idx in enumerate(self.bonds.bonds[i, 2:]):
                        if bonded_idx == torch.inf:
                            continue
                        start_pos = struct_positions[i]
                        end_pos = self.positions[self.monad_mask][
                            torch.where(
                                self.universal_monad_identifier ==
                                bonded_idx
                            )[0][0].long()
                        ]
                        pygame.draw.line(
                            screen,
                            colors["B"],
                            (
                                int(start_pos[0].item()),
                                int(start_pos[1].item())
                            ),
                            (
                                int(end_pos[0].item()),
                                int(end_pos[1].item())
                            ),
                            1
                        )

        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = self.colors[i]
            size = self.sizes[i].item()
            idx = self.from_general_to_monad_idx(i)

            if thing_type == "energyUnit":
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)
            elif thing_type == "monad":
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)
            elif thing_type == "structuralUnit":
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)

            if show_info and thing_type == "monad":
                # Show energy
                energy_text = self.energies[idx].item()
                if energy_text < 1000:
                    energy_text = str(int(energy_text))
                elif energy_text < 10000:
                    energy_text = f"{int(energy_text / 100) / 10:.1f}k"
                else:
                    energy_text = f"{int(energy_text / 1000)}k"
                energy_text = self.font.render(energy_text, True, colors["RGB"])
                energy_rect = energy_text.get_rect(
                    center = (
                        int(pos[0].item()),
                        int(pos[1].item() - 2 * size)
                    )
                )
                screen.blit(energy_text, energy_rect)

                # Show universal ID
                UID_text = self.universal_monad_identifier[idx].item()
                UID_text = self.font.render(UID_text, True, colors["RGB"])
                UID_rect = UID_text.get_rect(
                    center = (
                        int(pos[0].item()),
                        int(pos[1].item() + 2 * size)
                    )
                )
                screen.blit(UID_text, UID_rect)

            if show_sight and thing_type == "monad":
                draw_dashed_circle(screen, self.colors[i], (int(pos[0].item()),
                                   int(pos[1].item())), SIGHT)

    def get_state(self):
        return {
            'types': self.thing_types,
            'positions': self.positions.tolist(),
            'energies': self.energies.tolist(),
            'genomes': self.genomes.tolist(),
            'str_manipulations': self.str_manipulations.tolist(),
            'lineages': self.lineages,
            'colors': self.colors,
            'memory': self.memory.tolist(),
            'Rotation': self.Rotation.tolist(),
            'bonds': self.bonds.bonds.tolist(),
            'bond_sites': self.bond_sites.tolist(),
            'universalID': self.universal_monad_identifier.tolist(),
            'universalN': self.total_number_of_all_monads
        }

    def load_state(self, state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)["things_state"]

        self.thing_types = state['types']
        self.sizes = torch.tensor(
            [THING_TYPES[x]["size"] for x in self.thing_types]
        )
        self.positions = torch.tensor(state['positions'])
        self.energies = torch.tensor(state['energies'])
        self.N = len(self.positions)
        self.genomes = torch.tensor(state['genomes'])
        self.lineages = state['lineages']
        self.colors = state['colors']
        self.str_manipulations = torch.tensor(state['str_manipulations'])
        self.memory = torch.tensor(state['memory'])
        self.Rotation = torch.tensor(state['Rotation'])
        self.U = torch.stack(
            (
                torch.cos(self.Rotation),
                torch.sin(self.Rotation)
            ),
            dim = 1
        )
        self.universal_monad_identifier = torch.tensor(state['universalID'])
        self.total_number_of_all_monads = state['universalN']

        self.monad_mask = torch.tensor(
            [thing_type == "monad" for thing_type in self.thing_types]
        )
        self.energy_mask = torch.tensor(
            [thing_type == "energyUnit" for thing_type in self.thing_types]
        )
        self.structure_mask = torch.tensor(
            [thing_type == "structuralUnit" for thing_type in self.thing_types]
        )
        self.memory_mask = torch.tensor(
            [thing_type == "memoryUnit" for thing_type in self.thing_types]
        )
        self.Pop = self.monad_mask.sum().item()
        self.E = self.energies.sum().item() // 1000

        self.apply_genomes()

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)

        self.distances = None
        self.initialize_bonds()
        self.bonds.bonds = torch.tensor(state['bonds'])
        self.bond_sites = torch.tensor(state['bond_sites'])

    def add_structuralUnits(self, POP_STR = 1):
        self.thing_types += ["structuralUnit" for _ in range(POP_STR)]
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(
                    [THING_TYPES["structuralUnit"]["size"]
                     for _ in range(POP_STR)]
                )
            ),
            dim = 0
        )
        self.positions = add_positions(POP_STR, self.positions)
        self.colors += [THING_TYPES["structuralUnit"]["color"]
                        for _ in range(POP_STR)]
        self.N += POP_STR
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.zeros(POP_STR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.zeros(POP_STR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.ones(POP_STR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.memory_mask = torch.cat(
            (
                self.memory_mask,
                torch.zeros(POP_STR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.str_manipulations = torch.cat(
            (
                self.str_manipulations,
                torch.cat(
                    (
                        torch.zeros((POP_STR, 2, 1), dtype = torch.float32),
                        torch.rand(
                            (POP_STR, 2, 1),
                            dtype = torch.float32
                        ) * 20 - 10,
                        torch.zeros((POP_STR, 2, 1), dtype = torch.float32)
                    ),
                    dim = 2
                )
            ),
            dim = 0
        )

    def place_memoryUnits(self, POP_MMR = 1):
        self.thing_types += ["memoryUnit" for _ in range(POP_STR)]
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(
                    [THING_TYPES["memoryUnit"]["size"]
                     for _ in range(POP_MMR)]
                )
            ),
            dim = 0
        )
        self.positions = add_positions(POP_MMR, self.positions)
        self.colors += [THING_TYPES["memoryUnit"]["color"]
                        for _ in range(POP_MMR)]
        self.N += POP_MMR
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.zeros(POP_MMR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.zeros(POP_MMR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.zeros(POP_MMR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.memory_mask = torch.cat(
            (
                self.memory_mask,
                torch.ones(POP_MMR, dtype = torch.bool)
            ),
            dim = 0
        )

    def initialize_bonds(self):
        self.bonds = Bonds(self.structure_mask.sum().item())
