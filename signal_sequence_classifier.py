"""
Signal-to-Sequence Transformer
Deep learning classifier for 1D signal data with transformer architecture.

Key features:
- Multi-scale temporal convolutions
- Rotary position embeddings (RoPE)
- Attention pooling
- Focal loss for imbalanced classes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
from dataclasses import dataclass
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Model architecture
    conv_channels: list = None      # [32, 64, 128, 256]
    num_conv_blocks: list = None    # [2, 2, 3, 3]
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Training
    batch_size: int = 128
    num_epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.01
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    
    # Data
    max_signal_length: int = 1024
    num_classes: int = 20
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256]
        if self.num_conv_blocks is None:
            self.num_conv_blocks = [2, 2, 3, 3]

# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class Mish(nn.Module):
    """Smooth activation function"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm

class SqueezeExcitation(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            Mish(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = F.adaptive_avg_pool1d(x, 1)
        scale = self.fc(scale)
        return x * scale

# ============================================================================
# TEMPORAL CONVOLUTIONAL BACKBONE
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with squeeze-excitation"""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcitation(channels)
        self.mish = Mish()
    
    def forward(self, x):
        residual = x
        out = self.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.mish(out + residual)

class TemporalConvNet(nn.Module):
    """Multi-scale temporal CNN with residual connections"""
    def __init__(self, channels_list, num_blocks_list):
        super().__init__()
        self.in_channels = channels_list[0]
        
        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(1, self.in_channels, 15, padding=7, bias=False),
            nn.BatchNorm1d(self.in_channels),
            Mish()
        )
        
        # Stages with downsampling
        self.stages = nn.ModuleList()
        for i, (out_c, num_blocks) in enumerate(zip(channels_list, num_blocks_list)):
            stage = []
            
            # Downsample (except first stage)
            if i > 0:
                stage.append(nn.Sequential(
                    nn.Conv1d(self.in_channels, out_c, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(out_c),
                    Mish()
                ))
                self.in_channels = out_c
            
            # Residual blocks with increasing dilation
            for j in range(num_blocks):
                dilation = 2 ** (j % 3)  # Dilations: 1, 2, 4, 1, 2, 4, ...
                stage.append(ResidualBlock(out_c, dilation))
            
            self.stages.append(nn.Sequential(*stage))
        
        self.output_dim = channels_list[-1]
    
    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        return x  # (B, C, L')

# ============================================================================
# ROTARY POSITION EMBEDDINGS
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to input tensor"""
    d = x.shape[-1]
    x_rot = torch.cat([
        x[..., :d//2] * cos - x[..., d//2:] * sin,
        x[..., :d//2] * sin + x[..., d//2:] * cos
    ], dim=-1)
    return x_rot

# ============================================================================
# TRANSFORMER WITH ROPE
# ============================================================================

class TransformerBlock(nn.Module):
    """Transformer encoder with RoPE and pre-norm"""
    def __init__(self, d_model, nhead, dim_ff, max_len, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.rotary = RotaryEmbedding(d_model // nhead, max_len)
        
        # Feed-forward with SwiGLU
        self.ff_gate = nn.Linear(d_model, dim_ff)
        self.ff_up = nn.Linear(d_model, dim_ff)
        self.ff_down = nn.Linear(dim_ff, d_model)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm + RoPE attention
        x_norm = self.norm1(x)
        cos, sin = self.rotary(x.size(1), x.device)
        x_rot = apply_rotary_emb(x_norm, cos, sin)
        
        attn_out, _ = self.self_attn(
            x_rot, x_rot, x_rot,
            key_padding_mask=mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        # Pre-norm + SwiGLU
        x_norm = self.norm2(x)
        ff_out = F.silu(self.ff_gate(x_norm)) * self.ff_up(x_norm)
        ff_out = self.ff_down(ff_out)
        x = x + self.dropout(ff_out)
        
        return x

# ============================================================================
# ATTENTION POOLING
# ============================================================================

class AttentionPooling(nn.Module):
    """Multi-head attention pooling for sequence aggregation"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                Mish(),
                nn.Linear(dim // 2, 1)
            ) for _ in range(num_heads)
        ])
        self.fusion = nn.Linear(dim * num_heads, dim)
    
    def forward(self, x, mask):
        """
        x: (B, L, D)
        mask: (B, L) - True for padding positions
        """
        pooled = []
        for attn in self.attention_heads:
            scores = attn(x).squeeze(-1)  # (B, L)
            scores = scores.masked_fill(mask, -1e9)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
            pooled.append((x * weights).sum(dim=1))  # (B, D)
        
        pooled = torch.cat(pooled, dim=-1)  # (B, D * num_heads)
        return self.fusion(pooled)  # (B, D)

# ============================================================================
# MAIN MODEL
# ============================================================================

class SignalSequenceTransformer(nn.Module):
    """
    Signal-to-sequence classifier with CNN + Transformer
    
    Architecture:
    1. Multi-scale temporal CNN (local features)
    2. Transformer with RoPE (global context)
    3. Attention pooling
    4. Classification head
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Temporal CNN backbone
        self.conv = TemporalConvNet(config.conv_channels, config.num_conv_blocks)
        
        # Project to model dimension
        self.proj = nn.Sequential(
            nn.Conv1d(self.conv.output_dim, config.model_dim, 1),
            nn.BatchNorm1d(config.model_dim)
        )
        
        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.model_dim,
                config.num_heads,
                config.dim_feedforward,
                config.max_signal_length,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Attention pooling
        self.pool = AttentionPooling(config.model_dim, num_heads=4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.model_dim // 2),
            RMSNorm(config.model_dim // 2),
            Mish(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, config.num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, signal, mask):
        """
        Args:
            signal: (B, L) - raw signal
            mask: (B, L) - padding mask (True for padding)
        Returns:
            logits: (B, num_classes)
        """
        # CNN backbone
        x = self.conv(signal)  # (B, C, L')
        x = self.proj(x)  # (B, model_dim, L')
        
        # Adjust mask for downsampled sequence
        B, C, L_out = x.shape
        L_in = mask.shape[1]
        ratio = L_out / L_in
        non_padded_len = (~mask).sum(dim=1)
        new_len = (non_padded_len * ratio).long()
        mask_conv = torch.arange(L_out, device=mask.device).expand(B, -1) >= new_len.unsqueeze(1)
        
        # Transformer
        x = x.transpose(1, 2)  # (B, L', model_dim)
        for layer in self.transformer:
            x = layer(x, mask_conv)
        
        # Pool and classify
        x = self.pool(x, mask_conv)
        logits = self.classifier(x)
        
        return logits

# ============================================================================
# DATASET
# ============================================================================

class SignalDataset(Dataset):
    """Generic dataset for 1D signal classification"""
    
    def __init__(self, signals, labels, max_len):
        self.signals = signals
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        label = self.labels[idx]
        
        # Pad or crop
        orig_len = len(signal)
        if len(signal) < self.max_len:
            signal = np.pad(signal, (0, self.max_len - len(signal)))
        elif len(signal) > self.max_len:
            signal = signal[:self.max_len]
            orig_len = self.max_len
        
        # Create mask
        mask = np.zeros(self.max_len, dtype=bool)
        mask[orig_len:] = True
        
        return {
            'signal': torch.FloatTensor(signal),
            'label': torch.LongTensor([label])[0],
            'mask': torch.BoolTensor(mask)
        }

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def calculate_class_weights(labels, device):
    """Calculate balanced class weights"""
    class_counts = np.bincount(labels)
    total = len(labels)
    n_classes = len(class_counts)
    weights = [total / (n_classes * count) if count > 0 else 0.0 for count in class_counts]
    return torch.FloatTensor(weights).to(device)

def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training"):
        signal = batch['signal'].to(device)
        labels = batch['label'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(signal, mask)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average='macro', zero_division=0)
    )

def validate_epoch(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            signal = batch['signal'].to(device)
            labels = batch['label'].to(device)
            mask = batch['mask'].to(device)
            
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(signal, mask)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average='macro', zero_division=0)
    )

# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate model on synthetic data"""
    logger.info("=== Signal-to-Sequence Transformer Demo ===")
    
    # Generate synthetic multi-class signal data
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = 1000
    num_classes = 10
    signal_length = 512
    
    # Create signals with class-specific patterns
    signals = []
    labels = []
    
    for i in range(num_samples):
        class_id = i % num_classes
        
        # Base signal
        signal = np.random.randn(signal_length) * 0.1
        
        # Add class-specific frequency component
        freq = (class_id + 1) * 0.05
        t = np.arange(signal_length)
        signal += np.sin(2 * np.pi * freq * t) * (class_id + 1) / num_classes
        
        signals.append(signal)
        labels.append(class_id)
    
    signals = np.array(signals)
    labels = np.array(labels)
    
    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    config = Config(num_classes=num_classes, max_signal_length=signal_length)
    
    train_ds = SignalDataset(signals[train_idx], labels[train_idx], signal_length)
    test_ds = SignalDataset(signals[test_idx], labels[test_idx], signal_length)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    # Initialize model
    device = torch.device(config.device)
    model = SignalSequenceTransformer(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler()
    
    # Train for a few epochs
    logger.info("\nTraining...")
    for epoch in range(5):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        test_loss, test_acc, test_f1 = validate_epoch(model, test_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}/5")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2%}, F1: {train_f1:.2%}")
        logger.info(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2%}, F1: {test_f1:.2%}")
    
    logger.info("\n✓ Demo complete!")

if __name__ == "__main__":
    demo()
