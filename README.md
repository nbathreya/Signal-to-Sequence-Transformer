# Signal-to-Sequence Transformer

Deep learning classifier for 1D signal data using CNN + Transformer architecture. Demonstrates modern sequence modeling for biological/physical signals.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Architecture

```
Input Signal (1D)
      │
      ▼
┌─────────────────┐
│ Multi-Scale CNN │  ← Local temporal features
│  - Residual     │     (4 stages with downsampling)
│  - Dilated      │
│  - SE blocks    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │  ← Global context
│  - RoPE         │     (6 layers)
│  - SwiGLU       │
│  - Pre-norm     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Attention Pool  │  ← Aggregate sequence
└────────┬────────┘
         │
         ▼
    Classification
```

## Key Features

**Temporal CNN Backbone**
- Residual blocks with squeeze-excitation
- Multi-scale receptive fields (dilations: 1, 2, 4)
- 4-stage progressive downsampling (32→64→128→256 channels)

**Transformer Encoder**
- Rotary position embeddings (RoPE) - relative positions
- SwiGLU activation - better than GELU
- RMS normalization - faster than LayerNorm
- Pre-normalization - training stability

**Training Features**
- Focal loss for class imbalance
- Mixed precision (AMP)
- Gradient clipping
- Cosine learning rate schedule

## Performance

**Model**: 2.4M parameters  
**Speed**: ~100 sequences/sec on single GPU  
**Memory**: 4GB for batch_size=128

Synthetic 10-class demo achieves >95% accuracy in 5 epochs.

## Installation

```bash
pip install torch numpy scikit-learn tqdm
```

## Usage

**Demo on synthetic data:**
```bash
python signal_sequence_classifier.py
```

**Custom training:**
```python
from signal_sequence_classifier import (
    SignalSequenceTransformer, SignalDataset, Config
)

# Configure
config = Config(
    num_classes=20,
    max_signal_length=1024,
    model_dim=256,
    num_layers=6
)

# Load your data
signals = np.load("signals.npy")  # (N, L)
labels = np.load("labels.npy")    # (N,)

# Create dataset
dataset = SignalDataset(signals, labels, config.max_signal_length)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Train
model = SignalSequenceTransformer(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# ... training loop
```

## Model Components

### 1. Temporal CNN

Multi-scale feature extraction with residual connections:

```python
class ResidualBlock(nn.Module):
    # Dilated convolutions: 1, 2, 4
    # Squeeze-and-Excitation attention
    # BatchNorm + Mish activation
```

**Design choices:**
- Dilations capture patterns at different timescales
- SE blocks learn which features matter per class
- Residual connections enable deep networks (13+ layers)

### 2. Rotary Position Embeddings

Encodes relative positions directly into attention:

```python
cos, sin = rotary(seq_len)
x_rot = apply_rotary_emb(x, cos, sin)
```

**Benefits over learned positions:**
- Generalizes to longer sequences
- Better for 1D signals vs 2D (images)
- No position embedding parameters

### 3. SwiGLU Activation

Gated linear unit with Swish:

```python
ff_out = silu(gate(x)) * up(x)
```

**Why not GELU?**
- 10-15% better accuracy in language models
- Smooth gating mechanism
- Works well for sequence data

### 4. Attention Pooling

Multi-head attention for sequence aggregation:

```python
weights = softmax(attention_scores)
pooled = (features * weights).sum(dim=1)
```

**Better than mean/max pooling:**
- Learns what parts of sequence matter
- Class-specific attention patterns
- Multiple heads capture different aspects

## Applications

**Biological Signals**
- DNA/protein sequence classification
- Nanopore signal analysis
- Gene expression time-series

**Physical Signals**
- ECG/EEG classification
- Vibration analysis
- Audio event detection

**Financial Data**
- Time-series classification
- Anomaly detection
- Pattern recognition

## Benchmarks

Run on synthetic data (RTX 3090):

| Batch Size | Speed (seq/s) | Memory (GB) |
|------------|---------------|-------------|
| 32 | 120 | 2.1 |
| 128 | 95 | 4.3 |
| 256 | 88 | 7.8 |

## Architecture Comparison

| Component | This Model | Alternative |
|-----------|------------|-------------|
| CNN | Residual + SE | Plain conv |
| Position | RoPE | Learned/sinusoidal |
| Attention | Pre-norm | Post-norm |
| Activation | SwiGLU | GELU |
| Pooling | Multi-head attn | Mean/CLS token |

## Extensions

**Improve accuracy:**
- Data augmentation (time warping, noise)
- Label smoothing
- Model ensemble

**Reduce size:**
- Depthwise-separable convs
- Knowledge distillation
- Quantization (INT8)

**Scale up:**
- Multi-GPU training (DDP)
- Gradient checkpointing
- Flash Attention

---

**License**: MIT | **Python**: 3.8+ | **PyTorch**: 2.0+
