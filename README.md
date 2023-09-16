# latent-diffusion

An implementation of text-to-image via [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) by Rombach et. al with a focus on training on a TPU-v3-8 VM. Includes full training code for the VAE and DM with minimal changes from the original paper.

Non-cherry-picked generated outputs after 90k steps at batch size 1024, lr=1e-4, ~25 million img subset of laion2B-en:
![90k outputs](/misc/preds_90112.jpg)

Loss curve:
![loss curve](/misc/training_loss_curve.jpg)
