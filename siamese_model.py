import torch.nn as nn


class SiameseEncoder(nn.Module):
    """
    Encoder module for the Siamese Network. It uses a backbone
    (e.g., a pretrained CNN) to extract features from the input images,
    and then applies an adaptive average pooling
    followed by flattening to produce a fixed-size representation vector for
    each image.
    """

    def __init__(self, backbone):
        super(SiameseEncoder, self).__init__()
        self.backbone = backbone

        self.representation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        features = self.backbone(x)
        representation = self.representation(features)
        return representation


class SiameseModel(nn.Module):
    """
    Siamese Network model that takes three inputs (anchor, positive, negative)
    and produces their corresponding embeddings using the shared encoder.
    """
    def __init__(self, backbone):
        super(SiameseModel, self).__init__()
        self.encoder = SiameseEncoder(backbone)

    def forward(self, anchor, positive, negative):
        anchor_rep = self.encoder(anchor)
        positive_rep = self.encoder(positive)
        negative_rep = self.encoder(negative)
        return anchor_rep, positive_rep, negative_rep

    def get_embeddings(self, x):
        """Inference method to get embeddings for a single input."""
        return self.encoder(x)
