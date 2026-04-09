from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MANN_HIDDEN_DIM = 512
DEFAULT_GATING_HIDDEN_DIM = 32
DEFAULT_NUM_EXPERTS = 8


@dataclass(frozen=True)
class MANNModelConfig:
    x_main_dim: int
    x_gate_dim: int
    y_dim: int
    y_pose_dim: int
    y_root_dim: int
    y_future_dim: int = 0
    hidden_dim: int = DEFAULT_MANN_HIDDEN_DIM
    gating_hidden_dim: int = DEFAULT_GATING_HIDDEN_DIM
    num_experts: int = DEFAULT_NUM_EXPERTS

    @classmethod
    def from_data_spec(
        cls,
        spec,
        hidden_dim=DEFAULT_MANN_HIDDEN_DIM,
        gating_hidden_dim=DEFAULT_GATING_HIDDEN_DIM,
        num_experts=DEFAULT_NUM_EXPERTS,
    ):
        y_pose_dim = spec.y_pose_slice.stop - spec.y_pose_slice.start
        y_root_dim = spec.y_root_slice.stop - spec.y_root_slice.start
        y_future_dim = 0 if spec.y_future_slice is None else (spec.y_future_slice.stop - spec.y_future_slice.start)
        return cls(
            x_main_dim=spec.x_main_dim,
            x_gate_dim=spec.x_gate_dim,
            y_dim=spec.y_dim,
            y_pose_dim=y_pose_dim,
            y_root_dim=y_root_dim,
            y_future_dim=y_future_dim,
            hidden_dim=hidden_dim,
            gating_hidden_dim=gating_hidden_dim,
            num_experts=num_experts,
        )


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x_gate):
        x = F.elu(self.fc1(x_gate))
        x = F.elu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ExpertLinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_experts = int(num_experts)

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_experts, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for expert_index in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[expert_index], a=5 ** 0.5)
        fan_in = self.in_features
        bound = 1.0 / fan_in**0.5 if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, expert_weights):
        blended_weight = torch.einsum("bk,koi->boi", expert_weights, self.weight)
        blended_bias = torch.einsum("bk,ko->bo", expert_weights, self.bias)
        return torch.bmm(blended_weight, x.unsqueeze(-1)).squeeze(-1) + blended_bias


class MANN(nn.Module):
    def __init__(self, config: MANNModelConfig):
        super().__init__()
        self.config = config

        self.gating_network = GatingNetwork(
            input_dim=config.x_gate_dim,
            hidden_dim=config.gating_hidden_dim,
            num_experts=config.num_experts,
        )
        self.expert_fc1 = ExpertLinear(
            in_features=config.x_main_dim,
            out_features=config.hidden_dim,
            num_experts=config.num_experts,
        )
        self.expert_fc2 = ExpertLinear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            num_experts=config.num_experts,
        )
        self.expert_fc3 = ExpertLinear(
            in_features=config.hidden_dim,
            out_features=config.y_dim,
            num_experts=config.num_experts,
        )

    @classmethod
    def from_data_spec(
        cls,
        spec,
        hidden_dim=DEFAULT_MANN_HIDDEN_DIM,
        gating_hidden_dim=DEFAULT_GATING_HIDDEN_DIM,
        num_experts=DEFAULT_NUM_EXPERTS,
    ):
        config = MANNModelConfig.from_data_spec(
            spec,
            hidden_dim=hidden_dim,
            gating_hidden_dim=gating_hidden_dim,
            num_experts=num_experts,
        )
        return cls(config)

    def forward(self, x_main, x_gate, return_aux=False):
        expert_weights = self.gating_network(x_gate)
        hidden = F.elu(self.expert_fc1(x_main, expert_weights))
        hidden = F.elu(self.expert_fc2(hidden, expert_weights))
        y_pred = self.expert_fc3(hidden, expert_weights)

        if return_aux:
            return {
                "y_pred": y_pred,
                "expert_weights": expert_weights,
            }
        return y_pred

    def split_prediction(self, y_pred):
        y_pose_end = self.config.y_pose_dim
        y_root_end = y_pose_end + self.config.y_root_dim
        outputs = {
            "y_pose": y_pred[..., :y_pose_end],
            "y_root": y_pred[..., y_pose_end:y_root_end],
        }
        if self.config.y_future_dim > 0:
            outputs["y_future"] = y_pred[..., y_root_end:y_root_end + self.config.y_future_dim]
        return outputs


def mann_mse_loss(y_pred, y):
    return F.mse_loss(y_pred, y)
