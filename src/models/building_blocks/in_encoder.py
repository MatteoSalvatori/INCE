from typing import Optional

import torch
import torch.nn as nn

from src.models.brain_struct import *
from src.models.building_blocks.interaction_network import InteractionNetwork


class INEncoder(nn.Module):

    def __init__(self,
                 general_info: GeneralInfo,
                 common_info: CommonInfo,
                 input_graph_embedding: InputGraphEmbeddingInfo,
                 gnn_info: GNNEmbeddingInfo,
                 device: str):
        super(INEncoder, self).__init__()
        # Sanity checks
        assert_msg = "IN_CHANNEL_AGG={} is not allowed, only the following values are currently available: [{}, {}, {}]"
        assert gnn_info.in_channel_agg in [IN_CHANNEL_AGG_CONCAT, IN_CHANNEL_AGG_MLP, IN_CHANNEL_AGG_SUM], \
            assert_msg.format(gnn_info.in_channel_agg, IN_CHANNEL_AGG_CONCAT, IN_CHANNEL_AGG_MLP, IN_CHANNEL_AGG_SUM)

        # General info
        self.channels = gnn_info.in_channels
        self.channel_agg = gnn_info.in_channel_agg
        nodes_num = general_info.con_num + general_info.cat_num
        if input_graph_embedding.cls_num > 0:
            nodes_num += input_graph_embedding.cls_num
        a, b = zip(*[[i, j] for i in range(nodes_num) for j in range(nodes_num) if i != j])
        self.indexes = torch.tensor([list(a), list(b)]).to(device)

        # Parameters
        self.node_dropouts = nn.ModuleList()
        self.edge_dropouts = nn.ModuleList()
        self.gnns = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        for i in range(gnn_info.gnn_num):
            self.node_dropouts.append(nn.Dropout(p=common_info.dropout))
            if i > 0:
                self.edge_dropouts.append(nn.Dropout(p=common_info.dropout))
            # channels, mlp_output_size, mlp_hidden_size, mlp_layer_num, with_input_channels, with_edge_features
            self.gnns.append(InteractionNetwork(self.channels,
                                                common_info.latent_space_size,
                                                common_info.latent_space_size,
                                                gnn_info.in_mlp_deep,
                                                with_input_channels=(i > 0),
                                                with_edge_features=(i > 0)))
            self.node_norms.append(nn.LayerNorm([self.channels, nodes_num, common_info.latent_space_size]))
            if i < (gnn_info.gnn_num - 1):
                self.edge_norms.append(nn.LayerNorm([self.channels,
                                                     nodes_num * (nodes_num - 1),
                                                     common_info.latent_space_size]))
        if self.channels > 1 and self.channel_agg == IN_CHANNEL_AGG_MLP:
            self.channel_mlp = nn.Linear(self.channels * common_info.latent_space_size, common_info.latent_space_size)

    def forward_xai(self, nodes: torch.tensor, edges: Optional[torch.tensor] = None) -> torch.tensor:
        """

        :param nodes: shape = [batch_size, node_num, features_size]
        :param edges: shape = [batch_size, edge num, features_size]
        :return: shape = [batch_size, node_num, features_size]
        """
        batch_size = nodes.shape[0]
        node_num = nodes.shape[1]

        # Encoder GNN
        for i, gnn in enumerate(self.gnns):
            nodes = self.node_dropouts[i](nodes)
            if i > 0:
                self.edge_dropouts[i - 1](edges)
            nodes, edges = gnn(x=nodes, edge_index=self.indexes, edge_attr=edges)
            nodes = self.node_norms[i](nodes.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            if i < (len(self.gnns) - 1):
                edges = self.edge_norms[i](edges.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        if self.channels > 1:
            if self.channel_agg == IN_CHANNEL_AGG_MLP:
                nodes = nn.functional.relu(self.channel_mlp(nodes.permute(1, 2, 3, 0).reshape(batch_size,
                                                                                              node_num,
                                                                                              -1)))
            elif self.channel_agg == IN_CHANNEL_AGG_CONCAT:
                nodes = nodes.permute(1, 2, 3, 0).reshape(batch_size, node_num, -1)
            else:
                nodes = nodes.sum(dim=0)
        else:
            nodes = nodes.squeeze()
        return nodes, edges

    def forward(self, nodes: torch.tensor, edges: Optional[torch.tensor] = None) -> torch.tensor:
        """

        :param nodes: shape = [batch_size, node_num, features_size]
        :param edges: shape = [batch_size, edge num, features_size]
        :return: shape = [batch_size, node_num, features_size]
        """
        nodes, _ = self.forward_xai(nodes, edges)
        return nodes
