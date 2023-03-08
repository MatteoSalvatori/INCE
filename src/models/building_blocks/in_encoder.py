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
        if input_graph_embedding.use_cls > 0:
            nodes_num += input_graph_embedding.use_cls
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

    def auto_test_output_shape(self, con_size: int, cat_size: int, latent_size: int,
                               cls: bool, channels: int, channel_agg: str, device_str: str):
        batch = 12
        tot_nodes = con_size + cat_size
        tot_nodes += cls
        tot_edges = tot_nodes * (tot_nodes - 1)
        nodes = torch.rand((batch, tot_nodes, latent_size)).to(device_str)
        edges = torch.rand((batch, tot_edges, latent_size)).to(device_str)
        res = self.forward(nodes, edges)
        multiplier = channels if (channels > 1) and (channel_agg == IN_CHANNEL_AGG_CONCAT) else 1
        assert res.shape[0] == batch
        assert res.shape[1] == tot_nodes
        assert res.shape[2] == multiplier * latent_size

    def auto_test_structure(self, layers_num, in_deep_layer, latent_size, channels, channel_agg):
        if channels > 1 and channel_agg == IN_CHANNEL_AGG_MLP:
            assert len(list(self.children())) == 6
        else:
            assert len(list(self.children())) == 5

        # Dropout nodes
        assert len(list(self.children())[0]) == layers_num
        for l in list(self.children())[0]:
            assert isinstance(l, nn.Dropout)

        # Dropout Edges
        assert len(list(self.children())[1]) == (layers_num - 1)
        for l in list(self.children())[1]:
            assert isinstance(l, nn.Dropout)

        # IN
        assert len(list(self.children())[2]) == layers_num
        for i, l in enumerate(list(self.children())[2]):
            assert isinstance(l, InteractionNetwork)
            assert len(list(list(list(l.children())[0].children())[0].children())[0]) == 2*in_deep_layer
            assert list(list(list(list(l.children())[0].children())[0].children())[0][0].parameters())[0].shape[0] == latent_size
            n = 2 if i == 0 else 3
            assert list(list(list(list(l.children())[0].children())[0].children())[0][0].parameters())[0].shape[1] == n * latent_size

        # Norm nodes
        assert len(list(self.children())[3]) == layers_num
        for l in list(self.children())[3]:
            assert isinstance(l, nn.LayerNorm)

        # Norm Edges
        assert len(list(self.children())[4]) == (layers_num - 1)
        for l in list(self.children())[4]:
            assert isinstance(l, nn.LayerNorm)


if __name__ == "__main__":
    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    n_layers = 2
    cls = 1
    dev = 'cpu'
    in_deep = 4
    in_chs = 1
    in_ch_agg = IN_CHANNEL_AGG_CONCAT

    gen_info = GeneralInfo(con_num=cons, cat_num=cats, cat_degrees=deg)
    com_info = CommonInfo(latent_space_size=latent, dropout=0.0)
    ige_info = InputGraphEmbeddingInfo(use_cls=cls)
    in_info = GNNEmbeddingInfo(gnn_type=IN_GNN,
                               gnn_num=n_layers,
                               in_mlp_deep=in_deep,
                               in_channels=in_chs,
                               in_channel_agg=in_ch_agg,
                               gat_heads=2)

    ime = INEncoder(general_info=gen_info,
                    common_info=com_info,
                    input_graph_embedding=ige_info,
                    gnn_info=in_info,
                    device=dev).to(dev)

    # Test on the output_shape
    ime.auto_test_output_shape(cons, cats, latent, cls, in_chs, in_ch_agg, dev)

    # Test on the mlp structure
    ime.auto_test_structure(n_layers, in_deep, latent, in_chs, in_ch_agg)

    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    n_layers = 2
    cls = 3
    dev = 'cpu'
    in_deep = 4
    in_chs = 1
    in_ch_agg = IN_CHANNEL_AGG_CONCAT

    gen_info = GeneralInfo(con_num=cons, cat_num=cats, cat_degrees=deg)
    com_info = CommonInfo(latent_space_size=latent, dropout=0.0)
    ige_info = InputGraphEmbeddingInfo(use_cls=cls)
    in_info = GNNEmbeddingInfo(gnn_type=IN_GNN,
                               gnn_num=n_layers,
                               in_mlp_deep=in_deep,
                               in_channels=in_chs,
                               in_channel_agg=in_ch_agg,
                               gat_heads=2)

    ime = INEncoder(general_info=gen_info,
                    common_info=com_info,
                    input_graph_embedding=ige_info,
                    gnn_info=in_info,
                    device=dev).to(dev)

    # Test on the output_shape
    ime.auto_test_output_shape(cons, cats, latent, cls, in_chs, in_ch_agg, dev)

    # Test on the mlp structure
    ime.auto_test_structure(n_layers, in_deep, latent, in_chs, in_ch_agg)

    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    n_layers = 4
    cls = 0
    dev = 'cpu'
    in_deep = 2
    in_chs = 1
    in_ch_agg = IN_CHANNEL_AGG_CONCAT

    gen_info = GeneralInfo(con_num=cons, cat_num=cats, cat_degrees=deg)
    com_info = CommonInfo(latent_space_size=latent, dropout=0.0)
    ige_info = InputGraphEmbeddingInfo(use_cls=cls)
    in_info = GNNEmbeddingInfo(gnn_type=IN_GNN,
                               gnn_num=n_layers,
                               in_mlp_deep=in_deep,
                               in_channels=in_chs,
                               in_channel_agg=in_ch_agg,
                               gat_heads=2)

    ime = INEncoder(general_info=gen_info,
                    common_info=com_info,
                    input_graph_embedding=ige_info,
                    gnn_info=in_info,
                    device=dev).to(dev)

    # Test on the output_shape
    ime.auto_test_output_shape(cons, cats, latent, cls, in_chs, in_ch_agg, dev)

    # Test on the mlp structure
    ime.auto_test_structure(n_layers, in_deep, latent, in_chs, in_ch_agg)

    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 16
    n_layers = 4
    cls = 0
    dev = 'cpu'
    in_deep = 2
    in_chs = 4
    in_ch_agg = IN_CHANNEL_AGG_CONCAT
    gen_info = GeneralInfo(con_num=cons, cat_num=cats, cat_degrees=deg)
    com_info = CommonInfo(latent_space_size=latent, dropout=0.0)
    ige_info = InputGraphEmbeddingInfo(use_cls=cls)
    in_info = GNNEmbeddingInfo(gnn_type=IN_GNN,
                               gnn_num=n_layers,
                               in_mlp_deep=in_deep,
                               in_channels=in_chs,
                               in_channel_agg=in_ch_agg,
                               gat_heads=2)

    ime = INEncoder(general_info=gen_info,
                    common_info=com_info,
                    input_graph_embedding=ige_info,
                    gnn_info=in_info,
                    device=dev).to(dev)

    # Test on the output_shape
    ime.auto_test_output_shape(cons, cats, latent, cls, in_chs, in_ch_agg, dev)

    # Test on the mlp structure
    ime.auto_test_structure(n_layers, in_deep, latent, in_chs, in_ch_agg)

    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    n_layers = 4
    cls = 0
    dev = 'cpu'
    in_deep = 2
    in_chs = 2
    in_ch_agg = IN_CHANNEL_AGG_SUM

    gen_info = GeneralInfo(con_num=cons, cat_num=cats, cat_degrees=deg)
    com_info = CommonInfo(latent_space_size=latent, dropout=0.0)
    ige_info = InputGraphEmbeddingInfo(use_cls=cls)
    in_info = GNNEmbeddingInfo(gnn_type=IN_GNN,
                               gnn_num=n_layers,
                               in_mlp_deep=in_deep,
                               in_channels=in_chs,
                               in_channel_agg=in_ch_agg,
                               gat_heads=2)

    ime = INEncoder(general_info=gen_info,
                    common_info=com_info,
                    input_graph_embedding=ige_info,
                    gnn_info=in_info,
                    device=dev).to(dev)

    # Test on the output_shape
    ime.auto_test_output_shape(cons, cats, latent, cls, in_chs, in_ch_agg, dev)

    # Test on the mlp structure
    ime.auto_test_structure(n_layers, in_deep, latent, in_chs, in_ch_agg)

    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    n_layers = 4
    cls = 0
    dev = 'cpu'
    in_deep = 2
    in_chs = 2
    in_ch_agg = IN_CHANNEL_AGG_MLP

    gen_info = GeneralInfo(con_num=cons, cat_num=cats, cat_degrees=deg)
    com_info = CommonInfo(latent_space_size=latent, dropout=0.0)
    ige_info = InputGraphEmbeddingInfo(use_cls=cls)
    in_info = GNNEmbeddingInfo(gnn_type=IN_GNN,
                               gnn_num=n_layers,
                               in_mlp_deep=in_deep,
                               in_channels=in_chs,
                               in_channel_agg=in_ch_agg,
                               gat_heads=2)

    ime = INEncoder(general_info=gen_info,
                    common_info=com_info,
                    input_graph_embedding=ige_info,
                    gnn_info=in_info,
                    device=dev).to(dev)

    # Test on the output_shape
    ime.auto_test_output_shape(cons, cats, latent, cls, in_chs, in_ch_agg, dev)

    # Test on the mlp structure
    ime.auto_test_structure(n_layers, in_deep, latent, in_chs, in_ch_agg)
