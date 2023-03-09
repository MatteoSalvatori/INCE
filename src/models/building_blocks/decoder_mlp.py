import torch
import torch.nn as nn


class DecoderMLP(nn.Module):

    def __init__(self,
                 input_size: int,
                 latent_space_size: int,
                 output_size: int,
                 layer_num: int,
                 sequence_length: int):
        super(DecoderMLP, self).__init__()

        self.output_size = output_size
        self.hiddens = nn.ModuleList()
        for i in range(layer_num):
            self.hiddens.append(nn.Linear(latent_space_size if i > 0 else input_size, latent_space_size))
        self.output = (nn.Linear(latent_space_size, output_size) if sequence_length <= 1 else
                       MultiNodeOutputLayer(output_size, latent_space_size, sequence_length))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """

        :param x: shape = [batch_size, features_num=input_size]
        :return: shape = [batch_size, output_size]
        """
        for h in self.hiddens:
            x = torch.nn.functional.relu(h(x))
        return self.output(x)

    def predict_cls(self, x: torch.tensor) -> torch.tensor:
        """

        :param x: shape = [batch_size, features_num=input_size]
        :return: shape = [batch_size, output_size]
        """
        for h in self.hiddens:
            x = torch.nn.functional.relu(h(x))
        return x


class MultiNodeOutputLayer(nn.Module):

    def __init__(self, output_size: int, latent_space_size: int, sequence_length: int):
        super(MultiNodeOutputLayer, self).__init__()
        self.outputs = nn.ModuleList()
        for _ in range(sequence_length):
            self.outputs.append(nn.Linear(latent_space_size, output_size))

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.stack([o(x[:, i, :]) for i, o in enumerate(self.outputs)], dim=0)  # [node_num, batch_size, num_targets]
        return torch.permute(out, (1, 0, 2)).sum(dim=1)
