from collections import namedtuple

from src.common.constants import *

GeneralInfo = namedtuple('GeneralInfo',
                         'con_num, cat_num, cat_degrees')

CommonInfo = namedtuple('CommonInfo',
                        'latent_space_size, dropout')

InputGraphEmbeddingInfo = namedtuple('InputGraphEmbeddingInfo',
                                     'cls_num')

InputMLPEmbeddingInfo = namedtuple('InputMLPEmbeddingInfo',
                                   'input_embedding_con')

GNNEmbeddingInfo = namedtuple('GNNEmbeddingInfo',
                              'gnn_type, gnn_num, in_mlp_deep, in_channels, in_channel_agg, gat_heads')

TransformerEmbeddingInfo = namedtuple('EncoderTransformerInfo',
                                      'transformer_type, layer_num, heads_num')

DecoderInfo = namedtuple('DecoderInfo',
                         'class_num, decoder_deep')


def build_brain_structs(dataset_data, brain_params):
    # Sanity check
    graph_and_mlp_emb = (brain_params[INPUT_GRAPH_EMBEDDING][ENABLE] and
                         brain_params[INPUT_MLP_EMBEDDING][ENABLE])
    assert not graph_and_mlp_emb,\
        "Input Graph Embedding and Input MLP Embedding cannot be simultaneously enabled"

    # Wrapper to Python struct
    general_info = GeneralInfo(con_num=dataset_data.con_num,
                               cat_num=dataset_data.cat_num,
                               cat_degrees=dataset_data.cat_degrees)

    common_info = CommonInfo(latent_space_size=brain_params[COMMON][LATENT_SPACE_SIZE],
                             dropout=brain_params[COMMON][DROPOUT])

    input_graph_embedding = None
    if brain_params[INPUT_GRAPH_EMBEDDING][ENABLE]:
        input_graph_embedding = InputGraphEmbeddingInfo(
            cls_num=brain_params[INPUT_GRAPH_EMBEDDING][CLS_NUM])

    input_mlp_embedding = None
    if brain_params[INPUT_MLP_EMBEDDING][ENABLE]:
        input_mlp_embedding = InputMLPEmbeddingInfo(
            input_embedding_con=brain_params[INPUT_MLP_EMBEDDING][INPUT_EMBEDDING_CON])

    encoder_gnn_info = None
    if brain_params[GRAPH_EMBEDDING][ENABLE]:
        encoder_gnn_info = GNNEmbeddingInfo(gnn_type=brain_params[GRAPH_EMBEDDING][GNN_TYPE],
                                            gnn_num=brain_params[GRAPH_EMBEDDING][GNN_NUM],
                                            in_mlp_deep=brain_params[GRAPH_EMBEDDING][IN_MLP_DEEP],
                                            in_channels=brain_params[GRAPH_EMBEDDING][IN_CHANNELS],
                                            in_channel_agg=brain_params[GRAPH_EMBEDDING][IN_CHANNEL_AGG],
                                            gat_heads=brain_params[GRAPH_EMBEDDING][GAT_HEADS])
    encoder_transformer_info = None
    if brain_params[TRANSFORMER_EMBEDDING][ENABLE]:
        encoder_transformer_info = TransformerEmbeddingInfo(
            transformer_type=brain_params[TRANSFORMER_EMBEDDING][TRANSFORMER_TYPE],
            layer_num=brain_params[TRANSFORMER_EMBEDDING][TRANSFORMER_NUM],
            heads_num=brain_params[TRANSFORMER_EMBEDDING][TRANSFORMER_HEADS]
        )

    decoder_info = DecoderInfo(class_num=dataset_data.problem_size,
                               decoder_deep=brain_params[DECODER][DECODER_DEEP])

    return (general_info, common_info, input_graph_embedding, input_mlp_embedding,
            encoder_gnn_info, encoder_transformer_info, decoder_info)
