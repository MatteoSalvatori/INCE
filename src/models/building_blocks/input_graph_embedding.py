from typing import List

import torch
import torch.nn as nn


class InputGraphEmbedding(nn.Module):

    def __init__(self,
                 con_features_num: int,
                 cat_features_num: int,
                 cat_features_degrees: List[int],
                 latent_space_size: int,
                 use_cls: int):
        super(InputGraphEmbedding, self).__init__()
        self.con_features_num = con_features_num
        self.cat_features_num = cat_features_num
        self.use_cls = use_cls

        # cls embedding as in the Transformer case
        if self.use_cls > 0:
            cls = torch.Tensor(self.use_cls, latent_space_size)
            self.cls = nn.Parameter(nn.init.xavier_uniform_(cls))

        # Continuous Features embedding
        if self.con_features_num > 0:
            self.con_emb = nn.ModuleList()
            for _ in range(self.con_features_num):
                self.con_emb.append(nn.Linear(1, latent_space_size))

        # Categorical Features embedding
        if self.cat_features_num > 0:
            self.cat_emb = nn.ModuleList()
            for size in cat_features_degrees:
                self.cat_emb.append(nn.Embedding(size, latent_space_size))

    def forward(self, x_con: torch.tensor, x_cat: torch.tensor) -> torch.tensor:
        """

        :param x_con: shape = [batch_size, con_features_num]
        :param x_cat: shape = [batch_size, cat_features_num]
        :return: shape = [batch_size, features_num=con_features_num+cat_features_num, latent_space_size]
        """
        batch_size = x_cat.shape[0] if x_cat is not None else x_con.shape[0]
        final_list = []
        if self.use_cls > 0:
            final_list.append(self.cls.repeat(batch_size, 1, 1))

        if self.con_features_num > 0:
            con_feature_nodes = torch.permute(
                torch.stack([torch.nn.functional.relu(con_emb(x_con[:, i].unsqueeze(dim=1)))
                             for i, con_emb in enumerate(self.con_emb)], dim=0), (1, 0, 2))
            final_list.append(con_feature_nodes)

        if self.cat_features_num > 0:
            cat_feature_nodes = torch.permute(
                torch.stack([cat_emb(x_cat[:, i]) for i, cat_emb in enumerate(self.cat_emb)], dim=0), (1, 0, 2))
            final_list.append(cat_feature_nodes)

        return torch.cat(final_list, dim=1)

    def auto_test_output_shape(self, con_size: int, cat_size: int, cat_degrees: List[int],
                               latent_size: int, cls_num: int, device_str: str):
        batch = 12
        con_data = torch.rand((batch, con_size)).to(device_str)
        cat_data = torch.tensor([d-1 for d in cat_degrees], dtype=torch.long).to(device).repeat((batch, 1))
        res = self.forward(con_data, cat_data)
        assert res.shape[0] == batch
        assert res.shape[1] == ((con_size + cat_size + cls_num) if cls_num else (con_size + cat_size))
        assert res.shape[2] == latent_size

    def auto_test_structure(self, con_num, cat_num, cat_degrees, latent_size):
        if self.use_cls > 0:
            assert self.cls.shape[0] == self.use_cls
            assert self.cls.shape[1] == latent_size

        tot = 1 if ((con_num == 0) or (cat_num == 0)) else 2
        assert len(list(self.children())) == tot

        if con_num > 0:
            index = 0
            assert isinstance(list(self.children())[index], nn.ModuleList)
            assert len(list(self.children())[index]) == con_num
            for l in list(self.children())[index]:
                assert isinstance(l, nn.Linear)
                assert list(l.parameters())[0].shape[0] == latent_size
                assert list(l.parameters())[0].shape[1] == 1

        if cat_num > 0:
            index = 1 if con_num > 0 else 0
            assert isinstance(list(self.children())[index], nn.ModuleList)
            assert len(list(self.children())[index]) == cat_num
            for i, l in enumerate(list(self.children())[index]):
                assert isinstance(l, nn.Embedding)
                assert list(l.parameters())[0].shape[0] == cat_degrees[i]
                assert list(l.parameters())[0].shape[1] == latent_size


if __name__ == "__main__":
    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    cls = 3
    device = 'cpu'
    ige = InputGraphEmbedding(con_features_num=cons,
                              cat_features_num=cats,
                              cat_features_degrees=deg,
                              latent_space_size=latent,
                              use_cls=cls).to(device)

    # Test on the output_shape
    ige.auto_test_output_shape(cons, cats, deg, latent, cls, device)

    # Test on the mlp structure
    ige.auto_test_structure(cons, cats, deg, latent)

    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    cls = 1
    device = 'cpu'
    ige = InputGraphEmbedding(con_features_num=cons,
                              cat_features_num=cats,
                              cat_features_degrees=deg,
                              latent_space_size=latent,
                              use_cls=cls).to(device)

    # Test on the output_shape
    ige.auto_test_output_shape(cons, cats, deg, latent, cls, device)

    # Test on the mlp structure
    ige.auto_test_structure(cons, cats, deg, latent)

    cons = 0
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    cls = 1
    ige = InputGraphEmbedding(con_features_num=cons,
                              cat_features_num=cats,
                              cat_features_degrees=deg,
                              latent_space_size=latent,
                              use_cls=cls)

    # Test on the output_shape
    ige.auto_test_output_shape(cons, cats, deg, latent, cls, device)

    # Test on the mlp structure
    ige.auto_test_structure(cons, cats, deg, latent)

    cons = 2
    cats = 0
    deg = []
    latent = 10
    cls = 1
    ige = InputGraphEmbedding(con_features_num=cons,
                              cat_features_num=cats,
                              cat_features_degrees=deg,
                              latent_space_size=latent,
                              use_cls=cls)

    # Test on the output_shape
    ige.auto_test_output_shape(cons, cats, deg, latent, cls, device)

    # Test on the mlp structure
    ige.auto_test_structure(cons, cats, deg, latent)

    cons = 2
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    cls = 0
    ige = InputGraphEmbedding(con_features_num=cons,
                              cat_features_num=cats,
                              cat_features_degrees=deg,
                              latent_space_size=latent,
                              use_cls=cls)

    # Test on the output_shape
    ige.auto_test_output_shape(cons, cats, deg, latent, cls, device)

    # Test on the mlp structure
    ige.auto_test_structure(cons, cats, deg, latent)

    cons = 0
    cats = 3
    deg = [3, 2, 2]
    latent = 10
    cls = 0
    ige = InputGraphEmbedding(con_features_num=cons,
                              cat_features_num=cats,
                              cat_features_degrees=deg,
                              latent_space_size=latent,
                              use_cls=cls)

    # Test on the output_shape
    ige.auto_test_output_shape(cons, cats, deg, latent, cls, device)

    # Test on the mlp structure
    ige.auto_test_structure(cons, cats, deg, latent)

    cons = 2
    cats = 0
    deg = []
    latent = 10
    cls = 0
    ige = InputGraphEmbedding(con_features_num=cons,
                              cat_features_num=cats,
                              cat_features_degrees=deg,
                              latent_space_size=latent,
                              use_cls=cls)

    # Test on the output_shape
    ige.auto_test_output_shape(cons, cats, deg, latent, cls, device)

    # Test on the mlp structure
    ige.auto_test_structure(cons, cats, deg, latent)
