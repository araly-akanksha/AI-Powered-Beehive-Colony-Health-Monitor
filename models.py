# ============================================================
# models.py — Both deep learning models in one file
#
# MODELS (read top to bottom):
#   1. BaselineCNN   — 3 conv blocks + global average pooling
#   2. CRNN          — Same CNN + GRU + attention (extends BaselineCNN)
#   3. get_model()   — factory function to instantiate either model
#
# KEY INSIGHT: CRNN wraps the same CNN feature extractor.
# This means you can directly compare them as a fair ablation.
# ============================================================

import torch
import torch.nn as nn
import config


# ==============================================================
# MODEL 1 — BASELINE CNN
# ==============================================================

class BaselineCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for mel-spectrogram classification.

    Architecture:
      Input: (batch, 1, N_MELS, time_frames)   ← mel-spectrogram as grayscale image
        ↓
      Conv Block 1: Conv2d(1→32)   + BatchNorm + ReLU + MaxPool
      Conv Block 2: Conv2d(32→64)  + BatchNorm + ReLU + MaxPool
      Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool
        ↓
      Global Average Pooling  → (batch, 128)   ← collapses spatial dims
        ↓
      Dropout
        ↓
      Fully Connected → (batch, NUM_CLASSES)
        ↓
      Output: class logits (queen_present, queen_absent)

    ~300K parameters. Establishes baseline performance.
    """

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(BaselineCNN, self).__init__()

        # Three convolutional blocks — each doubles the feature maps
        # and halves the spatial resolution via MaxPool
        self.conv_blocks = nn.Sequential(

            # Block 1: 1 → 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # spatial: H/2, W/2
            nn.Dropout2d(0.1),

            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # spatial: H/4, W/4

            # Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # spatial: H/8, W/8
        )

        # Global Average Pooling: averages over all spatial positions
        # Reduces (batch, 128, H', W') → (batch, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x shape: (batch_size, 1, N_MELS, time_frames)
        Returns: logits of shape (batch_size, num_classes)
        """
        x = self.conv_blocks(x)       # → (batch, 128, H', W')
        x = self.global_avg_pool(x)   # → (batch, 128, 1, 1)
        x = self.classifier(x)        # → (batch, num_classes)
        return x

    def extract_features(self, x):
        """Extract CNN features without the classifier head. Used by CRNN."""
        x = self.conv_blocks(x)   # → (batch, 128, H', W')
        return x


# ==============================================================
# MODEL 2 — CRNN (CNN + Recurrent Neural Network)
# ==============================================================

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for mel-spectrogram classification.

    Extends BaselineCNN by replacing Global Average Pooling with a
    Bidirectional GRU that processes CNN feature maps over time.

    WHY: The CNN captures local frequency patterns in each time frame.
    The GRU captures how those patterns evolve over time — important
    for bee buzz patterns (queen piping is a temporal sequence).

    Architecture:
      Input: (batch, 1, N_MELS, time_frames)
        ↓
      [Same 3 Conv Blocks as BaselineCNN]  → (batch, 128, H', T')
        ↓
      Reshape: merge frequency dim into features
               (batch, T', 128 * H')
        ↓
      Linear projection → (batch, T', GRU_HIDDEN)
        ↓
      Bidirectional GRU (2 layers) → (batch, T', GRU_HIDDEN*2)
        ↓
      Attention Pooling → (batch, GRU_HIDDEN*2)
        ↓
      Dropout + FC → (batch, NUM_CLASSES)
    """

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CRNN, self).__init__()

        # Reuse the same CNN backbone as BaselineCNN
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After 3 MaxPool(2,2): N_MELS=128 → 16 frequency bins remain
        cnn_freq_bins   = config.N_MELS // 8          # = 16
        cnn_feature_dim = 128 * cnn_freq_bins          # = 2048

        # Project CNN features to GRU input size
        self.feature_proj = nn.Linear(cnn_feature_dim, config.GRU_HIDDEN)

        # Bidirectional GRU — reads the feature sequence forward AND backward
        self.gru = nn.GRU(
            input_size    = config.GRU_HIDDEN,
            hidden_size   = config.GRU_HIDDEN,
            num_layers    = config.GRU_LAYERS,
            batch_first   = True,
            bidirectional = config.BIDIRECTIONAL,
            dropout       = config.DROPOUT if config.GRU_LAYERS > 1 else 0,
        )

        # Attention: learn which time steps are most important
        gru_out_size = config.GRU_HIDDEN * (2 if config.BIDIRECTIONAL else 1)
        self.attention = nn.Linear(gru_out_size, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(gru_out_size, num_classes),
        )

    def forward(self, x):
        """
        x shape: (batch_size, 1, N_MELS, time_frames)
        Returns: logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # CNN feature extraction
        x = self.conv_blocks(x)                        # → (batch, 128, freq', time')
        freq_bins  = x.size(2)
        time_steps = x.size(3)

        # Reshape: treat time as sequence length
        # (batch, 128, freq', time') → (batch, time', 128*freq')
        x = x.permute(0, 3, 1, 2)                     # → (batch, time', 128, freq')
        x = x.reshape(batch_size, time_steps, -1)      # → (batch, time', 128*freq')

        # Project to GRU hidden size
        x = self.feature_proj(x)                       # → (batch, time', GRU_HIDDEN)
        x = torch.relu(x)

        # GRU over time sequence
        x, _ = self.gru(x)                             # → (batch, time', GRU_HIDDEN*2)

        # Attention pooling — weighted sum over time steps
        # Instead of just taking the last hidden state, we let the model
        # decide which time steps carry the most information
        attn_weights = torch.softmax(self.attention(x), dim=1)  # → (batch, time', 1)
        x = (x * attn_weights).sum(dim=1)             # → (batch, GRU_HIDDEN*2)

        # Classification
        x = self.classifier(x)                         # → (batch, num_classes)
        return x


# ==============================================================
# MODEL 3 — DANN (Domain-Adversarial Neural Network)
# ==============================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from Ganin et al. (2016).
    In the forward pass, it acts as an identity operator.
    In the backward pass, it scales gradients by -alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network for cross-hive generalisation.
    Wraps the CRNN feature extractor with a Gradient Reversal Layer (GRL)
    and an adversarial hive-classification head.

    Core Mechanism:
      - Health Head: Learns to detect queen presence/absence.
      - Hive Head: Predicts which physical hive the audio came from.
      - Gradient Reversal: Inverts gradients from the Hive Head (-alpha).
      -> Forces the feature extractor to purge hive-specific box resonance
         and microphone coloration, learning invariant bee health signatures.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, num_domains=None):
        super(DANN, self).__init__()
        if num_domains is None:
            num_domains = len(config.HIVE_SPLITS.get("train", [])) or 12

        self.num_classes = num_classes
        self.num_domains = num_domains

        # 1. Feature Extractor (identical to CRNN for fair comparison)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        cnn_freq_bins   = config.N_MELS // 8          # = 16
        cnn_feature_dim = 128 * cnn_freq_bins          # = 2048

        self.feature_proj = nn.Linear(cnn_feature_dim, config.GRU_HIDDEN)
        self.gru = nn.GRU(
            input_size    = config.GRU_HIDDEN,
            hidden_size   = config.GRU_HIDDEN,
            num_layers    = config.GRU_LAYERS,
            batch_first   = True,
            bidirectional = config.BIDIRECTIONAL,
            dropout       = config.DROPOUT if config.GRU_LAYERS > 1 else 0,
        )
        gru_out_size = config.GRU_HIDDEN * (2 if config.BIDIRECTIONAL else 1)
        self.attention = nn.Linear(gru_out_size, 1)

        # 2. Health Classification Head (Task-specific)
        self.health_classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(gru_out_size, num_classes),
        )

        # 3. Domain Classification Head (Adversarial Hive ID predictor)
        self.domain_classifier = nn.Sequential(
            nn.Linear(gru_out_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, num_domains),
        )

    def extract_features(self, x):
        """Extract bottleneck feature embedding from audio spectrogram."""
        batch_size = x.size(0)
        x = self.conv_blocks(x)
        time_steps = x.size(3)
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_steps, -1)
        x = torch.relu(self.feature_proj(x))
        x, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        embedding = (x * attn_weights).sum(dim=1)
        return embedding

    def forward(self, x, alpha=1.0, return_domain=False):
        """
        x shape: (batch_size, 1, N_MELS, time_frames)
        Returns:
            If return_domain=False: logits of shape (batch, num_classes)
            If return_domain=True:  (health_logits, domain_logits)
        """
        features = self.extract_features(x)
        health_logits = self.health_classifier(features)

        if return_domain:
            reversed_features = GradientReversalFunction.apply(features, alpha)
            domain_logits = self.domain_classifier(reversed_features)
            return health_logits, domain_logits

        return health_logits


# ==============================================================
# MODEL FACTORY
# ==============================================================

def get_model(name, num_classes=config.NUM_CLASSES):
    """
    Instantiate a model by name.

    Args:
        name (str): 'baseline_cnn', 'crnn', or 'dann'
        num_classes (int): number of output classes

    Returns: nn.Module

    Usage:
        model = get_model('dann')
    """
    name = name.lower().strip()

    if name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    elif name == "crnn":
        model = CRNN(num_classes=num_classes)
    elif name == "dann":
        model = DANN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: '{name}'. Choose 'baseline_cnn', 'crnn', or 'dann'.")

    # Print parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {name}  |  Trainable parameters: {params:,}")
    return model


# ==============================================================
# QUICK SANITY CHECK
# Run: python models.py
# ==============================================================

if __name__ == "__main__":
    import config

    print("=" * 55)
    print("Model Sanity Check — Forward Pass")
    print("=" * 55)

    # Dummy input: batch of 4 mel-spectrograms
    # Shape matches what BeeDataset returns: (batch, 1, N_MELS, time_frames)
    # time_frames ≈ WINDOW_SAMPLES / HOP_LENGTH = 32000 / 512 = 62
    time_frames = config.WINDOW_SAMPLES // config.HOP_LENGTH + 1
    dummy_input = torch.randn(4, 1, config.N_MELS, time_frames)
    print(f"\nInput shape : {tuple(dummy_input.shape)}")

    for model_name in ["baseline_cnn", "crnn", "dann"]:
        model  = get_model(model_name)
        output = model(dummy_input)
        if model_name == "dann":
            h_out, d_out = model(dummy_input, return_domain=True)
            print(f"{model_name:15} -> health: {tuple(h_out.shape)}, domain: {tuple(d_out.shape)}  OK")
        else:
            print(f"{model_name:15} -> output shape: {tuple(output.shape)}  OK")

    print("\nAll models OK -- forward pass complete.")
