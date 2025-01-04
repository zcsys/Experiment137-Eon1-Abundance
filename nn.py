import torch
import torch.nn.functional as F
from helpers import flattened_identity_matrix

def get_num_parameters(input_size, output_size, network):
    if network == "nn2":
        L1_size = input_size * 4
        L2_size = max(input_size, 2 * output_size)
        return (
            L1_size * (input_size + 1) +
            L2_size * (L1_size + 1) +
            output_size * (L2_size + 1)
        )

    elif network == "nn03":
        L1_size = input_size * 4
        L2_size = input_size * 3
        L3_size = input_size * 2
        return (
            L1_size * (input_size + 1) +
            L2_size * (L1_size + 1) +
            L3_size * (L2_size + 1) +
            output_size * (L3_size + 1)
        )

    elif network == "nn13":
        L1_size = input_size * 4
        L2_size = input_size * 3
        L3_size = input_size * 2
        return (
            L1_size * (input_size + 1) +
            L2_size * (input_size + L1_size + 1) +
            L3_size * (input_size + L1_size + L2_size + 1) +
            output_size * (input_size + L1_size + L2_size + L3_size + 1)
        )

    elif network == "nn23":
        L1_size = input_size * 8
        L2_size = input_size * 4
        L3_size = input_size * 2
        return (
            L1_size * (input_size + 1) +
            L2_size * (L1_size + 1) +
            L3_size * (L2_size + 1) +
            output_size * (L3_size + 1)
        )

def initialize_parameters(num_monads, input, output, network = "nn2"):
    if network == "nn2":
        L1_size = 4 * input
        L2_size = max(input, 2 * output)
        return torch.tensor(
            flattened_identity_matrix(input) +
            [0 for _ in range(input * (L1_size - input))] +
            [0 for _ in range(L1_size)] +
            flattened_identity_matrix(L1_size)[:L1_size * L2_size] +
            [0 for _ in range(L2_size)] +
            [0 for _ in range((L2_size + 1) * output)],
            dtype = torch.float32
        ).repeat(num_monads, 1)

    elif network == "nn03":
        L1_size = 4 * input
        L2_size = 3 * input
        L3_size = 2 * input
        return torch.tensor(
            flattened_identity_matrix(input) +
            [0 for _ in range(input * (L1_size - input))] +
            [0 for _ in range(L1_size)] +
            flattened_identity_matrix(L1_size)[:L1_size * L2_size] +
            [0 for _ in range(L2_size)] +
            flattened_identity_matrix(L2_size)[:L2_size * L3_size] +
            [0 for _ in range(L3_size)] +
            [0 for _ in range((L3_size + 1) * output)],
            dtype = torch.float32
        ).repeat(num_monads, 1)

    elif network == "nn13":
        return torch.zeros(
            (num_monads, get_num_parameters(input, output, network)),
            dtype = torch.float32
        )

class nn2:
    def __init__(self, weights, input_size, output_size):
        self.num_monads = weights.shape[0]
        L1_size = input_size * 4
        L2_size = max(input_size, 2 * output_size)

        num_parameters = get_num_parameters(input_size, output_size, "nn2")
        assert weights.shape[1] == num_parameters, "Weight size mismatch"

        pos = 0
        self.W1 = weights[
            :, :L1_size * input_size
        ].view(self.num_monads, L1_size, input_size)
        pos += L1_size * input_size
        self.B1 = weights[:, pos:pos + L1_size].unsqueeze(2)
        pos += L1_size

        self.W2 = weights[
            :, pos:pos + L2_size * L1_size
        ].view(self.num_monads, L2_size, L1_size)
        pos += L2_size * L1_size
        self.B2 = weights[:, pos:pos + L2_size].unsqueeze(2)
        pos += L2_size

        self.Wo = weights[
            :, pos:pos + output_size * L2_size
        ].view(self.num_monads, output_size, L2_size)
        pos += output_size * L2_size
        self.Bo = weights[:, pos:].unsqueeze(2)

    def forward(self, inputs):
        ff_1 = torch.relu(self.W1 @ inputs + self.B1)
        ff_2 = torch.relu(self.W2 @ ff_1 + self.B2)
        return torch.tanh(self.Wo @ ff_2 + self.Bo).squeeze(2)

class nn03:
    def __init__(self, weights, input_size, output_size):
        self.num_monads = weights.shape[0]
        L1_size = input_size * 4
        L2_size = input_size * 3
        L3_size = input_size * 2

        num_parameters = get_num_parameters(input_size, output_size, "nn03")
        assert weights.shape[1] == num_parameters, "Weight size mismatch"

        pos = 0
        self.W1 = weights[
            :, :L1_size * input_size
        ].view(self.num_monads, L1_size, input_size)
        pos += L1_size * input_size
        self.B1 = weights[:, pos:pos + L1_size].unsqueeze(2)
        pos += L1_size

        self.W2 = weights[
            :, pos:pos + L2_size * L1_size
        ].view(self.num_monads, L2_size, L1_size)
        pos += L2_size * L1_size
        self.B2 = weights[:, pos:pos + L2_size].unsqueeze(2)
        pos += L2_size

        self.W3 = weights[
            :, pos:pos + L3_size * L2_size
        ].view(self.num_monads, L3_size, L2_size)
        pos += L3_size * L2_size
        self.B3 = weights[:, pos:pos + L3_size].unsqueeze(2)
        pos += L3_size

        self.Wo = weights[
            :, pos:pos + output_size * L3_size
        ].view(self.num_monads, output_size, L3_size)
        pos += output_size * L3_size
        self.Bo = weights[:, pos:].unsqueeze(2)

    def forward(self, inputs):
        ff_1 = torch.relu(self.W1 @ inputs + self.B1)
        ff_2 = torch.relu(self.W2 @ ff_1 + self.B2)
        ff_3 = F.leaky_relu(self.W3 @ ff_2 + self.B3, negative_slope = 0.01)
        return torch.tanh(self.Wo @ ff_3 + self.Bo).squeeze(2)

class nn13:
    def __init__(self, weights, input_size, output_size):
        self.num_monads = weights.shape[0]
        L1_size = input_size * 4
        L2_size = input_size * 3
        L3_size = input_size * 2

        num_parameters = get_num_parameters(input_size, output_size, "nn13")
        assert weights.shape[1] == num_parameters, "Weight size mismatch"

        pos = 0
        self.W1 = weights[
            :, :L1_size * input_size
        ].view(self.num_monads, L1_size, input_size)
        pos += L1_size * input_size
        self.B1 = weights[:, pos:pos + L1_size].unsqueeze(2)
        pos += L1_size

        self.W2 = weights[
            :, pos:pos + L2_size * (input_size + L1_size)
        ].view(self.num_monads, L2_size, input_size + L1_size)
        pos += L2_size * (input_size + L1_size)
        self.B2 = weights[:, pos:pos + L2_size].unsqueeze(2)
        pos += L2_size

        self.W3 = weights[
            :, pos:pos + L3_size * (input_size + L1_size + L2_size)
        ].view(self.num_monads, L3_size, input_size + L1_size + L2_size)
        pos += L3_size * (input_size + L1_size + L2_size)
        self.B3 = weights[:, pos:pos + L3_size].unsqueeze(2)
        pos += L3_size

        self.Wo = weights[
            :, pos:pos + output_size * (input_size + L1_size + L2_size +
                                        L3_size)
        ].view(self.num_monads, output_size, input_size + L1_size + L2_size +
                                             L3_size)
        pos += output_size * (input_size + L1_size + L2_size + L3_size)
        self.Bo = weights[:, pos:].unsqueeze(2)

    def forward(self, inputs):
        ff_1 = torch.relu(self.W1 @ inputs + self.B1)

        cat_2 = torch.cat((inputs, ff_1), dim = 1)
        ff_2 = torch.relu(self.W2 @ cat_2 + self.B2)

        cat_3 = torch.cat((cat_2, ff_2), dim = 1)
        ff_3 = F.leaky_relu(self.W3 @ cat_3 + self.B3, negative_slope = 0.01)

        cat_o = torch.cat((cat_3, ff_3), dim = 1)
        return torch.tanh(self.Wo @ cat_o + self.Bo).squeeze(2)

class nn23:
    def __init__(self, weights, input_size, output_size):
        self.num_monads = weights.shape[0]
        L1_size = input_size * 8
        L2_size = input_size * 4
        L3_size = input_size * 2

        num_parameters = get_num_parameters(input_size, output_size, "nn23")
        assert weights.shape[1] == num_parameters, "Weight size mismatch"

        pos = 0
        self.W1 = weights[
            :, :L1_size * input_size
        ].view(self.num_monads, L1_size, input_size)
        pos += L1_size * input_size
        self.B1 = weights[:, pos:pos + L1_size].unsqueeze(2)
        pos += L1_size

        self.W2 = weights[
            :, pos:pos + L2_size * (input_size + L1_size)
        ].view(self.num_monads, L2_size, input_size + L1_size)
        pos += L2_size * (input_size + L1_size)
        self.B2 = weights[:, pos:pos + L2_size].unsqueeze(2)
        pos += L2_size

        self.W3 = weights[
            :, pos:pos + L3_size * (input_size + L1_size + L2_size)
        ].view(self.num_monads, L3_size, input_size + L1_size + L2_size)
        pos += L3_size * (input_size + L1_size + L2_size)
        self.B3 = weights[:, pos:pos + L3_size].unsqueeze(2)
        pos += L3_size

        self.Wo = weights[
            :, pos:pos + output_size * (input_size + L1_size + L2_size +
                                        L3_size)
        ].view(self.num_monads, output_size, input_size + L1_size + L2_size +
                                             L3_size)
        pos += output_size * (input_size + L1_size + L2_size + L3_size)
        self.Bo = weights[:, pos:].unsqueeze(2)

    def forward(self, inputs):
        ff_1 = torch.relu(self.W1 @ inputs + self.B1)

        cat_2 = torch.cat((inputs, ff_1), dim = 1)
        ff_2 = torch.relu(self.W2 @ cat_2 + self.B2)

        cat_3 = torch.cat((cat_2, ff_2), dim = 1)
        ff_3 = F.leaky_relu(self.W3 @ cat_3 + self.B3, negative_slope = 0.01)

        cat_o = torch.cat((cat_3, ff_3), dim = 1)
        return torch.tanh(self.Wo @ cat_o + self.Bo).squeeze(2)
