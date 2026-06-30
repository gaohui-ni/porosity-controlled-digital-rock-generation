import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        g = 8 if c >= 8 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(g, c),
            nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1),
            nn.GroupNorm(g, c),
            nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B,C,D,H,W]
        B, C, D, H, W = inputs.shape
        assert C == self.embedding_dim

        x = inputs.permute(0, 2, 3, 4, 1).contiguous()   # [B,D,H,W,C]
        flat_x = x.view(-1, self.embedding_dim)          # [N,C]

        emb = self.embeddings.weight                     # [K,C]
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(emb ** 2, dim=1) -
            2 * flat_x @ emb.t()
        )                                                 # [N,K]

        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        quantized = self.embeddings(encoding_indices).view(B, D, H, W, C)

        e_latent = F.mse_loss(quantized.detach(), x)
        q_latent = F.mse_loss(quantized, x.detach())
        vq_loss = q_latent + self.commitment_cost * e_latent

        # straight-through
        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()

        enc_oh = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = enc_oh.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity

class VQVAE256Down4Light(nn.Module):
    """
    down=4 only:
      256 -> 128 -> 64 (latent 64^3)
    max_ch is limited (<=96) to fit on 40GB.
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float,
                 base_ch: int = 32, max_ch: int = 96):
        super().__init__()
        C = int(embedding_dim)
        b = int(base_ch)
        m = int(max_ch)

        c1 = min(b, m)            # 32
        c2 = min(b * 2, m)        # 64
        c3 = min(b * 3, m)        # 96  (max)

        # Encoder: 256 -> 128 -> 64
        self.encoder = nn.Sequential(
            nn.Conv3d(1, c1, 4, stride=2, padding=1), nn.SiLU(),   # 256->128
            ResBlock3D(c1),
            nn.Conv3d(c1, c2, 4, stride=2, padding=1), nn.SiLU(),  # 128->64
            ResBlock3D(c2),
            nn.Conv3d(c2, c3, 3, padding=1), nn.SiLU(),
            ResBlock3D(c3),
            nn.Conv3d(c3, C, 3, padding=1),  # -> embedding_dim
        )

        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=C, commitment_cost=commitment_cost)

        # Decoder: 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.Conv3d(C, c3, 3, padding=1), nn.SiLU(),
            ResBlock3D(c3),

            nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1), nn.SiLU(),  # 64->128
            ResBlock3D(c2),

            nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1), nn.SiLU(),  # 128->256
            ResBlock3D(c1),

            nn.Conv3d(c1, max(16, c1 // 2), 3, padding=1), nn.SiLU(),
            nn.Conv3d(max(16, c1 // 2), 1, 3, padding=1),  # logits
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, ppl = self.vq(z)
        logits = self.decoder(z_q)
        prob = torch.sigmoid(logits)
        return prob, logits, vq_loss, ppl
