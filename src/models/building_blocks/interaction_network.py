from typing import Optional

import torch
from torch_geometric import nn as pyg_nn

from src.models.building_blocks.mlp import MLP


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class InteractionNetwork(pyg_nn.MessagePassing):
    """
    Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html

    Initial Code: https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing
    """
    def __init__(self,
                 channels: int,
                 mlp_output_size: int,
                 mlp_hidden_size: int,
                 mlp_layer_num: int,
                 with_input_channels: bool,
                 with_edge_features: bool):
        super().__init__()
        self.channels = channels
        self.with_input_channels = with_input_channels
        self.with_edge_features = with_edge_features

        self.lin_edge = torch.nn.ModuleList()
        self.lin_node = torch.nn.ModuleList()
        multiplier = 3 if with_edge_features else 2
        for ch in range(self.channels):
            self.lin_edge.append(MLP(mlp_hidden_size * multiplier, mlp_hidden_size, mlp_output_size, mlp_layer_num))
            self.lin_node.append(MLP(mlp_hidden_size * 2, mlp_hidden_size, mlp_output_size, mlp_layer_num))

    def forward(self, x, edge_index, edge_attr):
        if self.with_input_channels:
            assert len(x.shape) == 4
            assert x.shape[0] == self.channels
            if edge_attr is not None:
                assert edge_attr.shape[0] == self.channels
        else:
            x = x.repeat(self.channels, 1, 1, 1)
            if edge_attr is not None:
                edge_attr = edge_attr.repeat(self.channels, 1, 1, 1)
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_attr)
        node_out = torch.cat((x, aggr), dim=-1)
        node_out = torch.stack([l(node_out[ch, :, :, :])
                                for ch, l in enumerate(self.lin_node)], dim=0)
        if self.with_edge_features:
            edge_out = edge_attr + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        if self.with_edge_features:
            x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        else:
            x = torch.cat((x_i, x_j), dim=-1)
        x = torch.stack([l(x[ch, :, :, :])
                        for ch, l in enumerate(self.lin_edge)], dim=0)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        # out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        out = scatter_sum(inputs, index, dim=self.node_dim, dim_size=dim_size)
        return (inputs, out)
