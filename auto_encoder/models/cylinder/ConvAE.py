"""
@author: bochengz
@date: 2024/02/26
@email: bochengzeng@bochengz.top
"""
from auto_encoder.models import AutoEncoder
import torch
import torch.nn as nn


class ConvAE(AutoEncoder):
    """
    Built-in AutoEncoder, only consist of convolutional layers for cylinder \
    flow system

    Args:
        n_embed (int): the size of the embedding vector
        embed_drop (float): the drop rate of embedding
        layer_norm_eps (float): eps of LayerNorm for embedding
    """
    def __init__(self, n_embed, embed_drop, layer_norm_eps):
        super(ConvAE, self).__init__()

        X, Y = torch.meshgrid(torch.linspace(-2, 14, 128),
                              torch.linspace(-4, 4, 64), indexing='xy')
        self.mask = torch.sqrt(X ** 2 + Y ** 2) < 1
        self.n_embed = n_embed

        # Encoder conv. net
        self.observableNet = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), stride=2, padding=1,
                      padding_mode='replicate'),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16, 32, 64
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1,
                      padding_mode='replicate'),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32, 16, 32
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1,
                      padding_mode='replicate'),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64, 8, 16
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1,
                      padding_mode='replicate'),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128, 4, 8
            nn.Conv2d(128, n_embed // 32, kernel_size=(3, 3), stride=1,
                      padding=1, padding_mode='replicate'),
            # 4, 4, 8
        )

        self.observableNetFC = nn.Sequential(
            nn.LayerNorm(n_embed, eps=layer_norm_eps),
            nn.Dropout(embed_drop)
        )

        # Decoder conv. net
        self.recoveryNet = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_embed // 32, 128, kernel_size=(3, 3), stride=1,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 128, 8, 16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1,
                      padding_mode='replicate'),
            nn.ReLU(),
            # 64, 16, 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1,
                      padding_mode='replicate'),
            nn.ReLU(),
            # 32, 32, 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1,
                      padding_mode='replicate'),
            nn.ReLU(),
            # 16, 64, 128
            nn.Conv2d(16, 3, kernel_size=(3, 3), stride=1, padding=1,
                      padding_mode='replicate'),
            # 3, 64, 128
        )

        self.mu = torch.tensor([0., 0., 0., 0.])
        self.std = torch.tensor([1., 1., 1., 1.])

    def forward(self, x, visc):
        embedding = self.embed(x, visc)
        pred = self.recover(embedding)
        return embedding, pred

    def embed(self, x, visc):
        x = torch.cat(
            [x, visc.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:, :1])],
            dim=1)
        x = self._normalize(x)

        embedding = self.observableNet(x)
        embedding = self.observableNetFC(embedding.reshape(x.size(0), -1))
        return embedding

    def recover(self, embedding):
        x = self.recoveryNet(embedding.reshape(-1, self.n_embed // 32, 4, 8))
        x = self._unnormalize(x)
        # Apply cylinder mask
        mask0 = self.mask.repeat(x.size(0), x.size(1), 1, 1) == True
        x[mask0] = 0
        return x

    def apply_mu_std(self, mu, std, device):
        self.mu, self.std = mu.to(device), std.to(device)

    def _normalize(self, x):
        x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x

    def _unnormalize(self, x):
        return self.std[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1) * x + self.mu[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
