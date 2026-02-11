import torch.nn as nn


class InvertedBlock(nn.Module):
    """
    Inverted residual block with expansion and depthwise separable convolution.
    """

    def __init__(self, input_c, output_c, stride=1, expansion=1, shortcut=None):
        super().__init__()
        self.shortcut = shortcut
        self.hidden_c = int(input_c * expansion)

        self.expand = nn.Sequential(
            nn.Conv2d(input_c, self.hidden_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_c),
            nn.ReLU(),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                self.hidden_c,
                self.hidden_c,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=self.hidden_c,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_c),
            nn.ReLU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(self.hidden_c, output_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_c),
        )

    def forward(self, x):
        skip = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.shortcut is not None:
            skip = self.shortcut(x)

        out += skip

        return out


class Backbone(nn.Module):
    """
    A simple backbone classifier that uses a series of inverted blocks.
     - num_classes: Number of output classes for classification
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            self._make_block(32, 64, stride=2, expansion=3),
            self._make_block(64, 128, stride=2, expansion=3),
            self._make_block(128, 256, stride=2, expansion=3),
            self._make_block(256, 512, stride=2, expansion=3),
        )

    def _make_block(self, input_c, output_c, stride, expansion):
        condition = input_c != output_c or stride > 1

        shortcut = None
        if condition:
            shortcut = nn.Sequential(
                nn.Conv2d(input_c, output_c, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(output_c),
            )

        return InvertedBlock(
            input_c, output_c, stride=stride, expansion=expansion, shortcut=shortcut
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x


class Classifier(nn.Module):
    """
    A simple classifier that uses the backbone and a fully connected layer.
     - num_classes: Number of output classes for classification
    """

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x