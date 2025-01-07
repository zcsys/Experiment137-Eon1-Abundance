import torch
import random
import numpy as np
from base_vars import *
from scipy.spatial import KDTree

def unique(x):
    """Gets a list and returns its unique values as a list in same order"""
    seen = set()
    result = []
    for item in x:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def add_positions(N = 1,
                  existing_positions = torch.empty((0, 2)),
                  min_dist = 10.,
                  width = SIMUL_WIDTH,
                  height = SIMUL_HEIGHT):
    positions = existing_positions
    existing_N = len(positions)
    total_N = existing_N + N

    while len(positions) < total_N:
        remaining = total_N - len(positions)
        tree = KDTree(positions.numpy())
        new_pos = np.column_stack(
            (
                np.random.uniform(min_dist, width - min_dist, remaining),
                np.random.uniform(min_dist, height - min_dist, remaining)
            )
        )
        distances, _ = tree.query(new_pos)
        positions = torch.cat(
            (
                positions,
                torch.from_numpy(new_pos[distances >= min_dist]).float()
            ),
            dim = 0
        )

    return positions

def remove_element(tensor, i):
    return torch.cat((tensor[:i], tensor[i + 1:]), dim = 0)

def get_color_by_genome(genome, base_color = colors["rgb"]):
    n = len(genome) // 3
    return (
        min(max(base_color[0] + genome[:n].sum().int().item(), 0), 255),
        min(max(base_color[1] + genome[n:2 * n].sum().int().item(), 0), 255),
        min(max(base_color[2] + genome[2 * n:].sum().int().item(), 0), 255)
    )

def flattened_identity_matrix(N, x = None):
    lt = x if x else N
    return [1 if i == j and i < lt else 0 for j in range(N) for i in range(N)]

def vicinity(source_positions, radius = SIGHT, target_positions = None):
    source_tree = KDTree(source_positions.numpy())
    if target_positions:
        target_tree = KDTree(target_positions.numpy())
    else:
        target_tree, target_positions = source_tree, source_positions

    distances = source_tree.sparse_distance_matrix(target_tree, radius, p = 2.0)
    rows, cols = distances.nonzero()

    vector_diff = torch.zeros(
        (len(source_positions), len(target_positions), 2),
        dtype = torch.float32
    )
    vector_diff[rows, cols] = target_positions[cols] - source_positions[rows]

    return (
        torch.from_numpy(np.stack([rows, cols])),
        torch.tensor(distances.toarray(), dtype = torch.float32),
        vector_diff
    )

def decompose_vectors(X, U):
    U = U.unsqueeze(1)
    N, total_dims = X.shape
    X_reshaped = X.view(N, total_dims // 2, 2)
    parallel = torch.sum(X_reshaped * U, dim = 2, keepdim = True)
    perpendicular = torch.norm(X_reshaped - parallel * U, dim = 2)
    return torch.cat((parallel.squeeze(2), perpendicular), dim = 1)

def angle(pos0, pos1, pos2):
    vec1 = pos1 - pos0
    vec2 = pos2 - pos0

    vec1 /= torch.norm(vec1)
    vec2 /= torch.norm(vec2)

    return torch.acos(torch.dot(vec1, vec2))
