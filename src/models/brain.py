import torch
import torch.nn as nn

from src.models.brain_struct import *
from src.models.building_blocks.decoder_mlp import DecoderMLP
from src.models.building_blocks.input_graph_embedding import InputGraphEmbedding
from src.models.building_blocks.in_encoder import INEncoder


class Brain(nn.Module):

    def __init__(self,
                 general_info: GeneralInfo,
                 common_info: CommonInfo,
                 input_graph_embedding: InputGraphEmbeddingInfo,
                 input_mlp_embedding: InputMLPEmbeddingInfo,
                 encoder_gnn_info: GNNEmbeddingInfo,
                 decoder_info: DecoderInfo,
                 device: str):
        super(Brain, self).__init__()
        assert general_info.cat_num == len(general_info.cat_degrees), \
            "cat_num has to be compatible with the len of cat_degrees"
        assert not ((input_graph_embedding is not None) and (input_mlp_embedding is not None)), \
            "Only one between input_graph_embedding and input_mlp_embedding can be not None"
        assert not ((input_graph_embedding is None) and (input_mlp_embedding is None)),\
            "input_graph_embedding and input_mlp_embedding cannot be simultaneously None"

        # if (encoder_gnn_info is not None) or (encoder_transformer_info is not None):
        #     assert input_graph_embedding is not None, \
        #         "With Graph or Transformer encoder, input_graph_embedding is needed"

        self.continuous_num = general_info.con_num
        self.categorical_num = general_info.cat_num

        # ===========================================
        # ENCODER
        # ===========================================
        self.encoder = nn.ModuleList()

        # COLUMNAR EMBEDDING
        columnar_embedding = InputGraphEmbedding(con_features_num=general_info.con_num,
                                                 cat_features_num=general_info.cat_num,
                                                 cat_features_degrees=general_info.cat_degrees,
                                                 latent_space_size=common_info.latent_space_size,
                                                 use_cls=input_graph_embedding.use_cls)
        self.use_cls = input_graph_embedding.use_cls
        self.mlp_emb = False

        print("\tColumnar Embedding - Number of trainable parameters: {}"
              .format(sum(p.numel() for p in columnar_embedding.parameters() if p.requires_grad)))
        self.encoder.append(columnar_embedding)

        # CONTEXTUAL EMBEDDING - IN Graph ENCODER
        self.in_multichannel = False
        self.encoder.append(INEncoder(general_info=general_info,
                                      common_info=common_info,
                                      input_graph_embedding=input_graph_embedding,
                                      gnn_info=encoder_gnn_info,
                                      device=device))

        print("\tGNN Contextual Encoder - Number of trainable parameters: {}"
              .format(sum(p.numel() for p in self.encoder[-1].parameters() if p.requires_grad)))

        channels_multiplier = (1 if encoder_gnn_info.in_channel_agg != IN_CHANNEL_AGG_CONCAT else
                               encoder_gnn_info.in_channels)
        decoder_input_size = common_info.latent_space_size * self.use_cls * channels_multiplier
        decoder_sequence_length = 0

        # ===========================================
        # DECODER
        # ===========================================
        self.decoder = DecoderMLP(input_size=decoder_input_size,
                                  latent_space_size=common_info.latent_space_size,
                                  output_size=decoder_info.class_num,
                                  layer_num=decoder_info.decoder_deep,
                                  sequence_length=decoder_sequence_length)
        print("\tDecoder - Number of trainable parameters: {}"
              .format(sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)))

    def forward(self, features: torch.tensor) -> torch.tensor:
        """
        Forward function

        :param features: torch.tensor of shape = [batch_size, con_features_num + cat_features_num]

        :return torch.tensor of shape = [batch_size, num_targets]
        """
        # Encoder
        for i, layer in enumerate(self.encoder):
            if i == 0:
                con = features[:, :self.continuous_num] if self.continuous_num > 0 else None
                cat = features[:, self.continuous_num:].to(dtype=torch.long) if self.categorical_num > 0 else None
                features = layer(con, cat)
            else:
                features = layer(features)

        if not self.in_multichannel:
            batch_size = features.shape[0]
            x = (features[:, :self.use_cls, :].reshape(batch_size, -1)
                 if ((not self.mlp_emb) and self.use_cls > 0) else features)
        else:
            x = features

        # Decoder
        return self.decoder(x)  # shape = [batch_size, num_targets]
