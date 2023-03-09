from typing import List

import torch
import torch.nn as nn


class InputGraphEmbedding(nn.Module):

    def __init__(self,
                 con_features_num: int,
                 cat_features_num: int,
                 cat_features_degrees: List[int],
                 latent_space_size: int,
                 cls_num: int):
        super(InputGraphEmbedding, self).__init__()
        self.con_features_num = con_features_num
        self.cat_features_num = cat_features_num
        self.cls_num = cls_num

        # cls embedding as in the Transformer case
        if self.cls_num > 0:
            cls = torch.Tensor(self.cls_num, latent_space_size)
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
        if self.cls_num > 0:
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
