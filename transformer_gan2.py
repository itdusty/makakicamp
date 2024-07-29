import torch
import torch.nn as nn
import pandas as pd

from makakicamp.base import TorchGenerator
from makakicamp.gan import GANModule

from makakicamp.transformer_blocks2 import Classification, TransformerEncoderBlock


class GeneratorTransformerEncoderSequence(nn.Module):
    def __init__(self, depth: int = 5, **kwargs):
        super(GeneratorTransformerEncoderSequence, self).__init__()
        self.sequence = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.sequence:
            x = layer(x)
        return x


class DiscriminatorTransformerEncoderSequence(nn.Module):
    def __init__(self, depth: int = 5, **kwargs):
        super(DiscriminatorTransformerEncoderSequence, self).__init__()
        self.sequence = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.sequence:
            x = layer(x)
        return x


class TransformerDiscriminator(nn.Module):
    def __init__(self,
            sequence_length: int = 128,
            embedding_dim: int = 7,
            depth: int = 5,
            n_classes: int = 1,
            num_heads: int = 5,
            attention_dropout: float = 0.3,
            feed_dropout: float = 0.3,
        ):
        super(TransformerDiscriminator, self).__init__()
        self.disc = DiscriminatorTransformerEncoderSequence(
            depth=depth,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_dropout=attention_dropout,
            feed_dropout=feed_dropout
        )
        self.classifier = Classification(embedding_dim, sequence_length, n_classes)

    def forward(self, target: torch.tensor, condition: torch.tensor) -> torch.tensor:
        x = torch.cat([target, condition], dim=2)
        x = self.disc(x)
        return self.classifier(x), None

class TransformerGenerator(nn.Module):
    def __init__(self,
            sequence_length: int = 128,
            embedding_dim: int = 7,
            conditional_dim: int = 6,
            attention_dropout: float = 0.3,
            feed_dropout: float = 0.3,
            num_heads: int = 5,
            channels: int = 13,
            depth: int = 5
        ):
        super(TransformerGenerator, self).__init__()
        self.sequence_length = sequence_length
        self.attention_dropout = attention_dropout
        self.embedding_dim = embedding_dim
        self.conditional_dim = conditional_dim
        self.feed_dropout = feed_dropout
        self.num_heads = num_heads
        self.channels = channels

        #self.pos_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, self.embedding_dim))
        self.transformer_encoder = GeneratorTransformerEncoderSequence(
            depth=depth,
            embedding_dim=self.embedding_dim + self.conditional_dim,
            feed_dropout=self.feed_dropout,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout
        )
        self.first_linear = nn.Linear(self.sequence_length, self.sequence_length * self.embedding_dim)
        self.conv2d = nn.Conv2d(self.embedding_dim + self.conditional_dim, self.channels, kernel_size=1, padding=0)

    def forward(self, x: torch.tensor, condition: torch.tensor) -> torch.tensor:
        window_size = x.shape[1]
        x = x.view(-1, 1, window_size)
        x = self.first_linear(x).view(-1, self.sequence_length, self.embedding_dim)
        x = torch.cat([x, condition], dim=2)
        x = self.transformer_encoder(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.conv2d(x.permute(0, 3, 1, 2)).squeeze(2)
        output = output.reshape(-1, self.sequence_length, self.channels)
        return output

class TransformerGAN(TorchGenerator):
    def __init__(
            self,
            target_columns: list[str],
            conditional_columns: list[str],
            window_size: int = 10,
            batch_size: int = 16,
            num_epochs: int = 1,
            verbose: bool = False,
            embedding_dim: int = 7,
            channels: int = 7,
            num_heads_discriminator: int = 1,
            num_heads_generator: int = 1,
            num_disc_steps: int = 3,
            lr: float = 0.01,
        ):
        super().__init__(
            target_columns,
            conditional_columns,
            window_size,
            batch_size,
            num_epochs,
            verbose,
        )
        self.num_heads_discriminator = num_heads_discriminator
        self.num_heads_generator = num_heads_generator
        self.embedding_dim = embedding_dim
        self.conditional_dim = len(conditional_columns)
        self.channels = channels
        self.lr = lr
        self.num_disc_steps = num_disc_steps

    def fit(self, data: pd.DataFrame):
        gen = TransformerGenerator(
            sequence_length=self.window_size,
            embedding_dim=self.embedding_dim,
            channels=self.channels,
            attention_dropout=0.3,
            num_heads=self.num_heads_generator,
            depth=5,
            feed_dropout=0.3
        )
        disc = TransformerDiscriminator(
            sequence_length=self.window_size,
            embedding_dim=self.embedding_dim + self.conditional_dim,
            attention_dropout=0.3,
            num_heads=self.num_heads_discriminator,
            depth=5,
            feed_dropout=0.3
        )
        self.model = GANModule(
            gen,
            disc,
            latent_dim=1,
            lr=self.lr,
            num_disc_steps=self.num_disc_steps,
        )
        super().fit(data)

    def sample(self, data: pd.DataFrame, n_samples: int) -> list[pd.DataFrame]:
        last_i = 0
        output_data = []
        for i in range(self.window_size, data.shape[0], self.window_size):
            output_data.extend(super().sample(data[last_i:i], n_samples))
            last_i = i
        if data.shape[0] % self.window_size != 0 :
            tmp = super().sample(data[last_i - (self.window_size - data.shape[0] % self.window_size):], n_samples)
            for i in range(n_samples):
                tmp[i] = tmp[i][self.window_size - data.shape[0] % self.window_size:]
            output_data.extend(tmp)
        dct = {i:[] for i in range(n_samples)}
        for i in range(0, len(output_data)):
            dct[i % n_samples].extend(output_data[i].values.tolist())
        return [pd.DataFrame(data=val, columns=data.columns) for val in dct.values()]