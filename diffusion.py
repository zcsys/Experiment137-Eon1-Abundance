import torch
import torch.nn.functional as F
import pygame
import json
import numpy as np
from base_vars import *
from helpers import *

class Grid:
    def __init__(self, cell_size = GRID_CELL_SIZE, feature_dim = 3,
                 diffusion_rate = 0.001, saved_state = None):
        self.cell_size = cell_size
        self.feature_dim = feature_dim
        self.grid_x = SIMUL_WIDTH // cell_size
        self.grid_y = SIMUL_HEIGHT // cell_size
        self.diffusion_rate = diffusion_rate
        self.num_cells = self.grid_x * self.grid_y

        # Laplacian kernel
        self.kernel = torch.tensor(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ],
            dtype = torch.float32
        ).view(1, 1, 3, 3).repeat(feature_dim, 1, 1, 1)

        # Sobel kernels
        self.kernel_x = torch.tensor(
            [[-1, 0, 1]],
            dtype = torch.float32
        ).view(1, 1, 1, 3).repeat(self.feature_dim, 1, 1, 1)
        self.kernel_y = torch.tensor(
            [[-1], [0], [1]],
            dtype = torch.float32
        ).view(1, 1, 3, 1).repeat(self.feature_dim, 1, 1, 1)

        if saved_state:
            self.grid = torch.tensor(saved_state, dtype = torch.float32)
            self.fill()
            return

        # Initialize with correct dimensions (NCHW format)
        self.grid = torch.zeros((1, feature_dim, self.grid_y, self.grid_x),
                                dtype = torch.float32)
        self.fill()

    def add(self, channel, n, fill_target):
        y = torch.randint(0, self.grid_y, (n,))
        x = torch.randint(0, self.grid_x, (n,))
        self.grid[0, channel, y, x] = torch.max(
            self.grid[0, channel, y, x],
            torch.tensor(fill_target, dtype = torch.float32)
        )

    def fill(self, resource_per_cell = RESOURCE_TARGET,
             fill_target = FILL_TARGET):
        for i in range(self.feature_dim):
            tot = self.grid[0][i].sum()
            excess = (
                (tot - resource_per_cell * self.num_cells) //
                (fill_target - tot / self.num_cells)
            ).int()
            if excess < 0:
                self.add(i, -excess, fill_target)

    def gradient(self):
        self.padded = F.pad(self.grid, (1, 1, 1, 1), mode = "replicate")
        grad_x = F.conv2d(
            self.padded,
            self.kernel_x,
            padding = 0,
            groups = self.feature_dim
        )
        grad_y = F.conv2d(
            self.padded,
            self.kernel_y,
            padding = 0,
            groups = self.feature_dim
        )
        return grad_x, grad_y

    def diffuse(self, force_field = None):
        laplacian = F.conv2d(
            self.padded,
            self.kernel,
            padding = 0,
            groups = self.feature_dim
        )
        self.grid += self.diffusion_rate * laplacian
        if force_field is not None:
            self.apply_forces(force_field)
        torch.clamp_(self.grid[0], 0, 255)
        self.fill()

    def apply_forces(self, force_field, scale = 0.001):
        # Prevent overflows
        force_field.clamp_(-100, 100)

        # Get movements
        x_positive = torch.clamp(force_field[0], min = 0) * self.grid[0] * scale
        x_negative = torch.clamp(force_field[0], max = 0) * self.grid[0] * scale
        y_positive = torch.clamp(force_field[1], min = 0) * self.grid[0] * scale
        y_negative = torch.clamp(force_field[1], max = 0) * self.grid[0] * scale

        # Zero out edge movements
        x_positive[..., -1] = 0
        x_negative[..., 0] = 0
        y_positive[:, -1, :] = 0
        y_negative[:, 0, :] = 0

        # Apply movements
        self.grid[0] -= x_positive - x_negative + y_positive - y_negative
        self.grid[0] += x_positive.roll(1, dims = -1)  # Right
        self.grid[0] -= x_negative.roll(-1, dims = -1) # Left
        self.grid[0] += y_positive.roll(1, dims = -2)  # Down
        self.grid[0] -= y_negative.roll(-1, dims = -2) # Up

    def draw(self, surface):
        pygame.surfarray.blit_array(
            surface.subsurface((0, 0, SIMUL_WIDTH, SIMUL_HEIGHT)),
            torch.repeat_interleave(
                torch.repeat_interleave(
                    self.grid[0].permute(1, 2, 0),
                    self.cell_size,
                    dim = 0
                ),
                self.cell_size,
                dim = 1
            ).permute(1, 0, 2).numpy().astype(np.uint8)
        )
