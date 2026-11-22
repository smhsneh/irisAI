
import torch
import torch.nn as nn


class EarlyFusionHead(nn.Module):
    """
    Concatenates pooled image and text embeddings.
    Input dimensions: image=768, text=768.
    """

    def __init__(self, image_dim=768, text_dim=768, hidden_dim=512, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, image_pooled, text_pooled):
        fused = torch.cat([image_pooled, text_pooled], dim=1)
        return self.classifier(fused).view(-1)


class BilinearFusionHead(nn.Module):
    """
    Low-rank bilinear fusion using Hadamard interaction.
    """

    def __init__(
        self,
        image_dim=768,
        text_dim=768,
        fusion_dim=512,
        hidden_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.image_projection = nn.Linear(image_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image_pooled, text_pooled):
        image_features = self.image_projection(image_pooled)
        text_features = self.text_projection(text_pooled)

        interaction = image_features * text_features

        fused = torch.cat(
            [interaction, image_pooled, text_pooled],
            dim=1
        )

        return self.classifier(fused).view(-1)


class CrossAttentionFusionHead(nn.Module):
    """
    Text tokens query visual patch tokens through cross-attention.
    """

    def __init__(
        self,
        image_dim=1024,
        text_dim=768,
        fusion_dim=512,
        num_heads=8,
        dropout=0.3
    ):
        super().__init__()

        self.image_projection = nn.Linear(image_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.image_norm = nn.LayerNorm(image_dim)
        self.text_norm = nn.LayerNorm(text_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, image_tokens, text_tokens, attention_mask):
        image_features = self.image_projection(image_tokens)
        text_features = self.text_projection(text_tokens)

        # Text queries attend to image keys and values.
        attended, _ = self.cross_attention(
            query=text_features,
            key=image_features,
            value=image_features,
            need_weights=False
        )

        text_features = self.norm1(text_features + attended)
        text_features = self.norm2(
            text_features + self.feed_forward(text_features)
        )

        # Ignore padded text positions while pooling.
        mask = attention_mask.unsqueeze(-1).to(text_features.dtype)

        pooled_text = (
            (text_features * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        pooled_image = image_features.mean(dim=1)

        fused = torch.cat([pooled_image, pooled_text], dim=1)

        return self.classifier(fused).view(-1)


class TransformerFusionHead(nn.Module):
    """
    Joint transformer encoder over visual and textual token sequences.
    Uses modality embeddings to distinguish the two modalities.
    """

    def __init__(
        self,
        image_dim=1024,
        text_dim=768,
        fusion_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.image_projection = nn.Linear(image_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)

        self.image_modality_embedding = nn.Parameter(
            torch.randn(1, 1, fusion_dim) * 0.02
        )
        self.text_modality_embedding = nn.Parameter(
            torch.randn(1, 1, fusion_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, image_tokens, text_tokens, attention_mask):
        image_features = (
            self.image_projection(image_tokens)
            + self.image_modality_embedding
        )

        text_features = (
            self.text_projection(text_tokens)
            + self.text_modality_embedding
        )

        # Image tokens are all valid.
        # Text padding positions must be masked.
        combined = torch.cat(
            [image_features, text_features],
            dim=1
        )

        image_padding = torch.zeros(
            image_tokens.size(0),
            image_tokens.size(1),
            dtype=torch.bool,
            device=image_tokens.device
        )

        text_padding = ~attention_mask.bool()

        padding_mask = torch.cat(
            [image_padding, text_padding],
            dim=1
        )

        encoded = self.transformer(
            combined,
            src_key_padding_mask=padding_mask
        )

        encoded_image = encoded[:, :image_tokens.size(1), :]
        encoded_text = encoded[:, image_tokens.size(1):, :]

        pooled_image = encoded_image.mean(dim=1)

        text_mask = attention_mask.unsqueeze(-1).to(encoded_text.dtype)

        pooled_text = (
            (encoded_text * text_mask).sum(dim=1)
            / text_mask.sum(dim=1).clamp(min=1)
        )

        fused = torch.cat([pooled_image, pooled_text], dim=1)

        return self.classifier(fused).view(-1)