import math
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, layers: int, layernorm: bool =True):
        """ Multi-Layer perceptron

        Initial Code: https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing

        :param input_size: int,
        :param hidden_size: int
        :param output_size: int
        :param layers: int
        :param layernorm: bool
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
