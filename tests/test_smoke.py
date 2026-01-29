import torch

from src.model import ConvVAE, vae_loss_bce_logits


def test_forward_shapes_and_loss_is_finite():
    torch.manual_seed(0)

    model = ConvVAE(z_dim=32)
    model.eval()

    x = torch.rand(4, 1, 28, 28)  # Fashion-MNIST shape

    with torch.no_grad():
        logits, mu, logvar = model(x)

    assert logits.shape == (4, 1, 28, 28)
    assert mu.shape == (4, 32)
    assert logvar.shape == (4, 32)

    total, recon, kl = vae_loss_bce_logits(x=x, logits=logits, mu=mu, logvar=logvar, beta=1.0)
    assert torch.isfinite(total).item()
    assert torch.isfinite(recon).item()
    assert torch.isfinite(kl).item()


def test_decode_logits_shape():
    model = ConvVAE(z_dim=32)
    model.eval()

    z = torch.randn(5, 32)
    with torch.no_grad():
        logits = model.decode_logits(z)

    assert logits.shape == (5, 1, 28, 28)
