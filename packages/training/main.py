from packages.tokenisers.midi_tokeniser import MidiTokenizer
from packages.dataloaders.midi_dataloader import get_dataloader
from packages.models.transformer import MusicTransformerVAE
from packages.losses.transformer_losses import vae_loss
import torch

tokenizer = MidiTokenizer()
dataloader = get_dataloader('maestro-v3.0.0/2018', tokenizer, batch_size=4)

model = MusicTransformerVAE(tokenizer.vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    src = batch  # (batch_size, seq_len)
    tgt = src.clone()

    logits, mu, logvar = model(src, tgt[:, :-1])
    loss, _, _ = vae_loss(logits, tgt[:, 1:], mu, logvar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Train loss: {loss.item():.2f}")
