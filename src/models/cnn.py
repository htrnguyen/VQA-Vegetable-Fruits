import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Union, Optional
from torchvision.models import ResNet50_Weights


class CustomCNN(nn.Module):
    """Custom CNN architecture cho VQA"""

    def __init__(self, output_dim: int = 2048):
        super().__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 5
            nn.Conv2d(512, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        output = self.avgpool(features)
        return output.view(output.size(0), -1)


class CNNEncoder(nn.Module):
    """CNN Encoder cho bài toán VQA với tùy chọn pretrained và spatial features"""

    def __init__(
        self,
        output_dim: int = 2048,
        model_type: str = "resnet50",
        pretrained: bool = True,
        use_spatial: bool = False,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            output_dim: Kích thước vector đặc trưng đầu ra
            model_type: Loại model ('resnet50', 'custom')
            pretrained: Có sử dụng pretrained weights không (chỉ cho resnet)
            use_spatial: Có giữ lại spatial features không
            freeze_backbone: Có đóng băng backbone không
        """
        super().__init__()

        self.use_spatial = use_spatial
        self.output_dim = output_dim

        if model_type == "resnet50":
            # Sử dụng pretrained ResNet50
            resnet = models.resnet50(pretrained=pretrained)

            if use_spatial:
                # Bỏ avgpool và fc layers để lấy spatial features
                modules = list(resnet.children())[:-2]
                self.backbone = nn.Sequential(*modules)
                # Thêm conv layer để điều chỉnh số channels
                self.output_conv = nn.Conv2d(2048, output_dim, kernel_size=1)
            else:
                # Sử dụng global features
                if freeze_backbone and pretrained:
                    for param in resnet.parameters():
                        param.requires_grad = False
                resnet.fc = nn.Linear(resnet.fc.in_features, output_dim)
                self.backbone = resnet
                self.output_conv = nn.Identity()

        elif model_type == "custom":
            # Custom CNN nhẹ hơn
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, output_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            if not use_spatial:
                self.output_conv = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
                )
            else:
                self.output_conv = nn.Identity()

        # Thêm batch normalization
        self.batch_norm = (
            nn.BatchNorm2d(output_dim) if use_spatial else nn.BatchNorm1d(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 3, height, width]
        features = self.backbone(x)
        features = self.output_conv(features)
        features = self.batch_norm(features)
        return features
