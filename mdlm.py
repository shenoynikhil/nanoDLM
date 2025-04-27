"""Diffusion Language Models with Masked Diffusion Language Models (MDLM) (Sahoo et al. 2024)"""

import os
import numpy as np

import lightning.pytorch as L
import torch
from torch import Tensor
from models.dit import DiT


class MDLMConfig:
    # --- data ----
    data_dir: str = "data"
    dataset: str = "shakespeare_chat"
    device_type: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- dataloader ----
    train_samples_per_epoch: int = 1000
    block_size: int = 128
    batch_size: int = 128

    # --- network ----
    vocab_size: int = 100
    n_layers: int = 2
    n_heads: int = 2
    hidden_dim: int = 64
    qkv_bias: bool = True

    # --- training ----
    noise_schedule: str = "log_linear"
    noise_eps: float = 1e-3
    mask_token: int = 0
    neg_infinity: float = -1e9
    sigma_max: float = 1e8
    sampling_steps: int = 1000 # number of diffusion steps (during inference)
    sampling_eps: float = 1e-5

    # --- sampling ----
    sample_and_print: int = 20

def decode(l, itos) -> str:
    return ''.join([itos[i] for i in l])


def _sample_categorical(categorical_probs: Tensor) -> Tensor:
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


class Dataset(torch.utils.data.Dataset):
    """Parts taken directly from poor man's get_batch from nanoGPT"""
    def __init__(self, split: str, cfg: MDLMConfig):
        super().__init__()
        dataset_dir = os.path.join(cfg.data_dir, cfg.dataset)
        if split == 'train':
            data = np.memmap(os.path.join(dataset_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(dataset_dir, 'val.bin'), dtype=np.uint16, mode='r')
        self.data = data
        self.cfg = cfg
        self.meta = pickle.load(open(os.path.join(dataset_dir, 'meta.pkl'), 'rb'))
        self.itos = self.meta['itos']
    
    def __len__(self):
        """Just to use the pytorch dataset, but we don't need this..."""
        return self.cfg.train_samples_per_epoch

    def __getitem__(self, idx):
        ix = np.random.randint(0, len(self.data) - self.cfg.block_size)
        item_np = self.data[ix:ix+self.cfg.block_size].astype(np.int64)
        return torch.from_numpy(item_np)


class MDLM(L.LightningModule):
    def __init__(self, cfg: MDLMConfig, itos: dict):
        super().__init__()
        self.cfg = cfg
        self.denoiser_network = DiT(
            vocab_size=cfg.vocab_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            hidden_dim=cfg.hidden_dim,
            qkv_bias=cfg.qkv_bias
        )

        self.itos = itos

    def sigma_and_dsigma_t(self, t: Tensor) -> Tensor:
        """noise schedule parameterization. Appendix E.1"""
        if self.cfg.noise_schedule == "linear":
            sigma_t = self.cfg.sigma_max * t
            dsigma_t = self.cfg.sigma_max * torch.ones_like(t)
        elif self.cfg.noise_schedule == "log_linear":
            factor = 1 - self.cfg.noise_eps # for numerical stability
            sigma_t = -torch.log(1 - factor * t)
            dsigma_t = factor / (1 - factor * t)
        else:
            raise NotImplementedError(f"Noise schedule {self.cfg.noise_schedule} not implemented")
        
        return sigma_t, dsigma_t

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """x: input, c: conditioning (usually the noise level)"""
        # compute the loss
        logits = self.denoiser_network(x, c) # [batch_size, seq_len, vocab_size]
    
        # SUBS parameterization 
        logits[:, :, self.cfg.mask_token] = self.cfg.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True) # normalize the logits in log space        
        unmasked_indices = (x != self.cfg.mask_token)
        logits[unmasked_indices] = self.cfg.neg_infinity # set all the unmasked token logits to -infinity
        logits[unmasked_indices, x[unmasked_indices]] = 0 # set the true token logit to 0    
        
        return logits

    def compute_loss(self, x0: Tensor) -> Tensor:
        # x0: [batch_size, num_tokens]
        # t = torch.rand((x0.shape[0], 1), device=x0.device) # t ~ U[0, 1]
        t = self.low_discrepancy_sample_t(x0.shape[0])
        sigma_t, dsigma_t = self.sigma_and_dsigma_t(t)
        alpha_t = torch.exp(-sigma_t)
        alpha_t_bar = -1 * alpha_t * dsigma_t

        # mask the tokens with a probability of 1 - alpha_t
        p_mask = torch.repeat_interleave((1 - alpha_t), x0.shape[1], dim=1)
        mask = torch.bernoulli(p_mask)
        xt = torch.where(mask == 1, self.cfg.mask_token, x0) # replace the mask positions with the mask token

        # compute the loss
        logits = self.forward(xt, c=sigma_t)
        
        # get log_p_theta for loss
        log_p_theta = torch.gather(logits, dim=-1, index=x0[:, :, None]).squeeze(-1) # [batch_size, seq_len]
        weight = -alpha_t_bar / (1 - alpha_t) # [batch_size,], difference of -1 from the paper works, TODO: check
        loss = -log_p_theta * weight # [batch_size, seq_len]
        loss = loss.mean()
        return loss

    def low_discrepancy_sample_t(self, n: int) -> Tensor:
        """Section 3.5.1: Sample time steps from a low-discrepancy sampler
        based on Variational diffusion models from Kingma et al. 2021
        """
        eps_t = torch.rand(n, device=self.device)
        offset = torch.arange(n, device=self.device) / n
        t = (eps_t / n + offset) % 1 # modulo 1 to avoid t > 1
        t = (1 - self.cfg.sampling_eps) * t + self.cfg.sampling_eps
        t = t[:, None] # [n, 1]
        return t

    def sample(
        self,
        num_samples: int = 1,
        num_steps: int | None = None,
        block_size: int | None = None,
        eps: float = 1e-5
    ):
        """Sample from the generative process.
        
        Args:
            num_samples: number of samples to sample.
            num_steps: number of diffusion steps to sample. If None, use the number of sampling steps from the config.
            eps: final time-step, used for setting the diffusion time-step schedule.
        """
        # prior is initially all mask tokens
        block_size = block_size if block_size is not None else self.cfg.block_size
        x = torch.ones(num_samples, block_size, device=self.device, dtype=torch.int64) * self.cfg.mask_token
        num_steps = num_steps if num_steps is not None else self.cfg.sampling_steps
        time_steps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps # time step size
        for i in range(num_steps):
            t = time_steps[i].unsqueeze(0)
            t = torch.repeat_interleave(t, num_samples, dim=0)[:, None] # [batch_size, 1]
            sigma_t, _ = self.sigma_and_dsigma_t(t)
            sigma_s, _ = self.sigma_and_dsigma_t(t - dt)
            alpha_t, alpha_s = torch.exp(-sigma_t), torch.exp(-sigma_s)

            # compute forward pass
            logits = self.forward(x, c=sigma_t) # [batch_size, seq_len, vocab_size]

            # construct the posterior distribution q(x_s | x_t, x_theta(x_t, c_t))
            # following Equation 7
            p_x0 = (alpha_s - alpha_t) / (1 - alpha_t)
            assert p_x0.shape == (x.shape[0], 1) # TODO: remove if passes
            q_xs = logits.exp() * p_x0[:, :, None] # [batch_size, seq_len, vocab_size]
            p_mask = (1 - alpha_s) / (1 - alpha_t)
            q_xs[:, :, self.cfg.mask_token] = p_mask[0]

            # sample from the posterior 
            x_sampled = _sample_categorical(q_xs) # [batch_size, seq_len]

            # keep the unmasked tokens from the previous step
            copy_unmasked = (x != self.cfg.mask_token)
            x = torch.where(copy_unmasked, x, x_sampled) # [batch_size, seq_len]
            
        return x

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(x0=batch)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self):
        if self.current_epoch % self.cfg.sample_and_print == 0:
            samples = self.sample(num_samples=2, block_size=64).cpu().numpy()
            for i, sample in enumerate(samples):
                s = decode(sample, self.itos)
                print (f'At epoch {self.current_epoch}, sample {i}: {s}')


if __name__ == "__main__":
    import pickle

    # Load meta info for vocab size
    with open('data/shakespeare_chat/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    # Set up config using meta info
    cfg = MDLMConfig()
    cfg.vocab_size = meta['vocab_size'] + 1 # +1 for the mask token
    cfg.data_dir = 'data'
    cfg.dataset = 'shakespeare_chat'
    cfg.mask_token = meta['vocab_size']
    cfg.batch_size = 128  # or any small number for testing
    cfg.block_size = 384  # or any value <= the length of your data
    cfg.hidden_dim = 256
    cfg.n_heads = 4
    cfg.n_layers = 4
    cfg.train_samples_per_epoch = 10000
    cfg.sample_and_print = 10

    # Create and move model to device
    model = MDLM(cfg, itos=meta['itos'])
    # do not save checkpoints
    trainer = L.Trainer(
        max_epochs=100, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False
    )
    train_loader = torch.utils.data.DataLoader(Dataset('train', cfg), batch_size=cfg.batch_size, pin_memory=True)
    trainer.fit(model, train_loader)
