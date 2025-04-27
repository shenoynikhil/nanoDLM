"""Diffusion Language Models with Masked Diffusion Language Models (MDLM) (Sahoo et al. 2024)"""

import os
# add current directory to sys.path
import sys
sys.path.append(os.path.dirname((os.path.abspath(__file__)))) # to allow importing dit
import numpy as np

import lightning.pytorch as L
import torch
from torch import Tensor
from dit import DiT


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
    mask_token: int = 0
    neg_infinity: float = -1e9
    sigma_max: float = 1e8
    T: int = 500 # number of diffusion steps (during inference)
    

# def get_batch(split: str, cfg: MDLMConfig):
#     """Poor man's dataloader acc to Karpathy (but extremely intuitive :))"""
#     # We recreate np.memmap every batch to avoid a memory leak, as per
#     # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
#     dataset_dir = os.path.join(cfg.data_dir, cfg.dataset)
#     if split == 'train':
#         data = np.memmap(os.path.join(dataset_dir, 'train.bin'), dtype=np.uint16, mode='r')
#     else:
#         data = np.memmap(os.path.join(dataset_dir, 'val.bin'), dtype=np.uint16, mode='r')
#     ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
#     x = torch.stack([torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64)) for i in ix])
#     if cfg.device_type == 'cuda':
#         device = torch.device("cuda")
#         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#         x = x.pin_memory().to(device, non_blocking=True)
#     else:
#         device = torch.device("cpu")
#         x = x.to(device)
#     return x


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
    
    def __len__(self):
        """Just to use the pytorch dataset, but we don't need this..."""
        return self.cfg.train_samples_per_epoch

    def __getitem__(self, idx):
        ix = np.random.randint(0, len(self.data) - self.cfg.block_size)
        item_np = self.data[ix:ix+self.cfg.block_size].astype(np.int64)
        return torch.from_numpy(item_np)


class MDLM(L.LightningModule):
    def __init__(self, cfg: MDLMConfig):
        super().__init__()
        self.cfg = cfg
        self.denoiser_network = DiT(
            vocab_size=cfg.vocab_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            hidden_dim=cfg.hidden_dim,
            qkv_bias=cfg.qkv_bias
        )
        self.noise_schedule = self.cfg.noise_schedule
        self.T = self.cfg.T

    def sigma_and_dsigma_t(self, t: Tensor) -> Tensor:
        """noise schedule parameterization. Appendix E.1"""
        if self.noise_schedule == "linear":
            sigma_t = self.cfg.sigma_max * t
            dsigma_t = self.cfg.sigma_max * torch.ones_like(t)
        elif self.noise_schedule == "log_linear":
            sigma_t = -torch.log1p(-t)
            dsigma_t = 1 / (1 - t)
        else:
            raise NotImplementedError(f"Noise schedule {self.noise_schedule} not implemented")
        
        return sigma_t, dsigma_t

    def forward(self, x: Tensor, sigma_t: Tensor) -> Tensor:
        # compute the loss
        logits = self.denoiser_network(x, sigma_t) # [batch_size, seq_len, vocab_size]

        # SUBS parameterization
        logits[:, :, self.cfg.mask_token] = self.cfg.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True) # normalize the logits in log space        
        unmasked_indices = (x != self.cfg.mask_token)
        logits[unmasked_indices] = self.cfg.neg_infinity # set all the unmasked token logits to -infinity
        logits[unmasked_indices, x[unmasked_indices]] = 0 # set the true token logit to 0

        return logits

    def compute_loss(self, x: Tensor) -> Tensor:
        # x: [batch_size, num_tokens]
        t = torch.rand((x.shape[0], 1), device=x.device) # t ~ U[0, 1]
        sigma_t, dsigma_t = self.sigma_and_dsigma_t(t)
        alpha_t = torch.exp(-sigma_t)
        alpha_t_bar = -1 * alpha_t * dsigma_t

        # mask the tokens with a probability of 1 - alpha_t
        p_mask = torch.repeat_interleave((1 - alpha_t), x.shape[1], dim=1)
        mask = torch.bernoulli(p_mask)
        xt = torch.where(
            mask == 1, torch.full_like(x, self.cfg.mask_token), x
        ) # replace the mask positions with the mask token

        # compute the loss
        logits = self.forward(x, sigma_t)
        log_p_theta = torch.gather(logits, dim=-1, index=xt[:, :, None]).squeeze(-1) # [batch_size, seq_len]

        weight = alpha_t_bar / (1 - alpha_t) # [batch_size,], difference of -1 from the paper works, TODO: check
        loss = -log_p_theta * weight[:, None] # [batch_size, seq_len]
        loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    import pickle

    # Load meta info for vocab size
    with open('data/shakespeare_chat/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    # Set up config using meta info
    cfg = MDLMConfig()
    cfg.vocab_size = meta['vocab_size']
    cfg.data_dir = 'data'
    cfg.dataset = 'shakespeare_chat'
    cfg.batch_size = 128   # or any small number for testing
    cfg.block_size = 128  # or any value <= the length of your data
    cfg.hidden_dim = 32
    cfg.n_heads = 8
    cfg.n_layers = 5
    cfg.train_samples_per_epoch = 10000

    # Create and move model to device
    model = MDLM(cfg)
    trainer = L.Trainer(max_epochs=10, accelerator="auto", devices="auto", logger=False)
    ds = Dataset('train', cfg)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, pin_memory=True)

    trainer.fit(model, train_loader)
