import os
import sys
import pytest
import torch
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdlm import MDLM, MDLMConfig, Dataset

# Check if data files exist
data_dir = os.path.join('data', 'shakespeare_chat')
meta_path = os.path.join(data_dir, 'meta.pkl')
train_path = os.path.join(data_dir, 'train.bin')
val_path = os.path.join(data_dir, 'val.bin')

# Skip all tests if data files don't exist
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(path) for path in [meta_path, train_path, val_path]),
    reason="Data files not found"
)

@pytest.fixture(scope="module")
def model_and_data():
    """Set up model and data for testing."""
    # Load meta info for vocab size
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Set up config using meta info
    cfg = MDLMConfig()
    cfg.vocab_size = meta['vocab_size']
    cfg.data_dir = 'data'
    cfg.dataset = 'shakespeare_chat'
    cfg.batch_size = 2
    cfg.block_size = 32
    cfg.hidden_dim = 32
    cfg.n_heads = 4
    cfg.n_layers = 2
    
    # Create model
    model = MDLM(cfg)
    device = torch.device(cfg.device_type)
    model = model.to(device)
    ds = Dataset('train', cfg)
    
    # Get a batch from the real data
    x = ds[0].unsqueeze(0) # add batch
    x = x.to(device)
    
    return model, x

def test_model_initialization(model_and_data):
    """Test that the model initializes correctly."""
    model = model_and_data[0]
    
    assert model is not None

def test_compute_loss(model_and_data):
    """Test that the compute_loss function works correctly."""
    model, x = model_and_data
    
    # Compute loss
    loss = model.compute_loss(x)
    
    # Check that loss is a scalar tensor
    assert loss.dim() == 0
    
    # Check that loss is not NaN or Inf
    assert not torch.isnan(loss).item()
    assert not torch.isinf(loss).item()
    
    # Check that loss is positive
    assert loss.item() > 0
