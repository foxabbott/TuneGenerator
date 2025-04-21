import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicTransformerVAE(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, latent_dim=256, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Improved embedding with layer normalization
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Encoder with improved configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Increased feedforward dimension
            dropout=dropout,
            activation='gelu',  # Using GELU activation
            batch_first=True  # More efficient memory layout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Improved latent space projection
        self.to_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim)
        )
        self.to_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim)
        )

        # Decoder with improved configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Improved output projection
        self.latent_to_embedding = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

    def encode(self, src):
        x = self.embedding(src)
        x = self.embedding_norm(x)
        x = x + self.pos_encoding[:src.shape[1]]
        memory = self.encoder(x)
        pooled = memory.mean(dim=1)  # Global average pooling
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, tgt, z):
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.embedding_norm(tgt_emb)
        tgt_emb = tgt_emb + self.pos_encoding[:tgt.shape[1]]
        
        latent_context = self.latent_to_embedding(z).unsqueeze(1).repeat(1, tgt.shape[1], 1)
        out = self.decoder(tgt_emb, latent_context)
        return self.output(out)

    def forward(self, src, tgt):
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(tgt, z)
        return logits, mu, logvar
