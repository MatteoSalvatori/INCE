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

    def auto_test_output_shape(self, input_size: int, output_size: int, seq_length: int, device: str):
        batch = 10
        if seq_length > 1:
            data = torch.rand((batch, seq_length, input_size)).to(device)
        else:
            data = torch.rand((batch, input_size)).to(device)
        res = self.forward(data)
        assert res.shape[0] == batch
        assert res.shape[1] == output_size

    def auto_test_structure(self, input_size, latent_size, output_size, layer_num, sequence_length):
        assert len(list(self.children())) == 2
        assert isinstance(list(self.children())[0], nn.ModuleList)
        assert len(list(self.children())[0]) == layer_num
        for i, l in enumerate(list(list(self.children())[0].children())):
            assert isinstance(l, nn.Linear)
            assert list(l.parameters())[0].shape[0] == latent_size
            assert list(l.parameters())[0].shape[1] == (input_size if i == 0 else latent_size)
        if sequence_length == 1:
            assert isinstance(list(self.children())[1], nn.Linear)
            assert list(list(self.children())[1].parameters())[0].shape[0] == output_size
            assert list(list(self.children())[1].parameters())[0].shape[1] == latent_size
        else:
            assert isinstance(list(self.children())[1], MultiNodeOutputLayer)


class MultiNodeOutputLayer(nn.Module):

    def __init__(self, output_size: int, latent_space_size: int, sequence_length: int):
        super(MultiNodeOutputLayer, self).__init__()
        self.outputs = nn.ModuleList()
        for _ in range(sequence_length):
            self.outputs.append(nn.Linear(latent_space_size, output_size))

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.stack([o(x[:, i, :]) for i, o in enumerate(self.outputs)], dim=0)  # [node_num, batch_size, num_targets]
        return torch.permute(out, (1, 0, 2)).sum(dim=1)

    def auto_test_output_shape(self, input_size: int, output_size: int, seq_lenght: int, device: str):
        batch = 10
        data = torch.rand((batch, seq_lenght, input_size)).to(device)
        res = self.forward(data)
        assert res.shape[0] == batch
        assert res.shape[1] == output_size

    def auto_test_structure(self, input_size, output_size, sequence_length):
        assert len(list(self.children())) == 1
        assert isinstance(list(self.children())[0], nn.ModuleList)
        assert len(list(self.children())[0]) == sequence_length
        for l in list(self.children())[0]:
            assert isinstance(l, nn.Linear)
            assert list(l.parameters())[0].shape[0] == output_size
            assert list(l.parameters())[0].shape[1] == input_size


if __name__ == "__main__":
    in_size = 10
    latent = 20
    out_size = 3
    seq_len = 5
    n_layers = 2
    mnol = MultiNodeOutputLayer(out_size, latent, seq_len)

    # Test on the output_shape
    mnol.auto_test_output_shape(input_size=latent, output_size=out_size, seq_lenght=seq_len, device='cpu')

    # Test on the mlp structure
    mnol.auto_test_structure(latent, out_size, seq_len)

    dmlp = DecoderMLP(input_size=in_size,
                      latent_space_size=latent,
                      output_size=out_size,
                      layer_num=n_layers,
                      sequence_length=seq_len)

    # Test on the output_shape
    dmlp.auto_test_output_shape(input_size=in_size, output_size=out_size, seq_length=seq_len, device='cpu')

    # Test on the mlp structure
    dmlp.auto_test_structure(in_size, latent, out_size, n_layers, seq_len)

    seq_len = 1
    dmlp = DecoderMLP(input_size=in_size,
                      latent_space_size=latent,
                      output_size=out_size,
                      layer_num=n_layers,
                      sequence_length=seq_len)

    # Test on the output_shape
    dmlp.auto_test_output_shape(input_size=in_size, output_size=out_size, seq_length=seq_len, device='cpu')

    # Test on the mlp structure
    dmlp.auto_test_structure(in_size, latent, out_size, n_layers, seq_len)
