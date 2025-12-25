import os
import torch
import matplotlib.pyplot as plt


def save_image_grid(tensor: torch.Tensor, path: str, n: int = 8, title: str | None = None):
    """
    Save an n-by-n grid of images.

    tensor: shape (B,1,H,W) expected, values in [0,1] preferred.
    We clamp to [0,1] to keep visualization stable.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    x = tensor.detach().cpu().clamp(0.0, 1.0)
    k = min(x.size(0), n * n)

    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for i in range(n * n):
        ax = axes[i // n, i % n]
        ax.axis("off")
        if i < k:
            ax.imshow(x[i, 0], cmap="gray")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)

def save_recon_pairs_grid(x: torch.Tensor, x_hat: torch.Tensor, path: str, cols: int = 8, pair_rows: int = 4, title: str | None = None):
    """
    Saves a grid of ORIGINAL vs RECON pairs.

    Layout:
      For each pair-row r:
        row 2r   = originals
        row 2r+1 = reconstructions

    - cols: number of images per row
    - pair_rows: number of (original,recon) row-pairs
      => total pairs shown = cols * pair_rows
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    x = x.detach().cpu().clamp(0.0, 1.0)
    x_hat = x_hat.detach().cpu().clamp(0.0, 1.0)

    pairs = min(x.size(0), cols * pair_rows)

    fig, axes = plt.subplots(2 * pair_rows, cols, figsize=(cols, 2 * pair_rows))
    for i in range(2 * pair_rows):
        for j in range(cols):
            axes[i, j].axis("off")

    for p in range(pairs):
        r = p // cols
        c = p % cols
        axes[2 * r, c].imshow(x[p, 0], cmap="gray")       # original
        axes[2 * r + 1, c].imshow(x_hat[p, 0], cmap="gray")  # recon

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_latent_interpolation(model, x_batch, device, out_path, steps: int = 12):
    """
    Saves a 1xN row showing interpolation between two inputs in latent space.
    Uses mu (deterministic) rather than sampled z to reduce noise.
    """
    # Set model to evaluation mode
    model.eval()

    x = x_batch.to(device)
    x_a = x[0:1] # (1, 1, 28, 28)
    x_b = x[1:2]

    mu_a, _ = model.encode(x_a) # (1, z_dim)
    mu_b, _ = model.encode(x_b)

    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1) # (steps, 1)
    z = (1.0 - alphas) * mu_a + alphas * mu_b # (steps, z_dim)

    logits = model.decode_logits(z) # (steps, 1, 28, 28)
    imgs = torch.sigmoid(logits).clamp(0.0, 1.0)

    # Save as a single row grid without assuming square grids
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, steps, figsize=(steps, 1), squeeze=False)
    for i in range(steps):
        axes[0, i].axis("off")
        axes[0,i].imshow(imgs[i, 0].detach().cpu(), cmap="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
