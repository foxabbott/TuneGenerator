import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(logits, targets, mu, logvar, pad_token_id=0, beta=0.1, kl_annealing_steps=10000, current_step=None):
    # Cross-entropy with label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
    ce_loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    # KL divergence with annealing
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = torch.mean(kl)
    
    # Free bits: ensure minimum KL divergence
    kl = torch.max(kl, torch.tensor(0.5).to(kl.device))
    
    # Anneal beta if current_step is provided
    if current_step is not None:
        current_step = torch.tensor(current_step, device=kl.device)
        annealed_beta = beta * min(1.0, current_step / kl_annealing_steps)
    else:
        annealed_beta = beta
    
    total_loss = ce_loss + annealed_beta * kl
    
    return total_loss, ce_loss.item(), kl.item()
