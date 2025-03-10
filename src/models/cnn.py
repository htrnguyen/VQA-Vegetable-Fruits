import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Union


class CNNEncoder(nn.Module):
    """CNN Encoder cho bài toán VQA với tùy chọn spatial features"""

    def __init__(
        self, output_dim: int = 512, pretrained: bool = True, use_spatial: bool = False
    ):
        """
        Args:
            output_dim: Kích thước vector đặc trưng đầu ra
            pretrained: Có sử dụng pretrained weights không
            use_spatial: Có giữ lại spatial features không
        """
        super(CNNEncoder, self).__init__()

        if pretrained:
            # Sử dụng pretrained ResNet-50
            resnet = models.resnet50(pretrained=True)

            if use_spatial:
                # Bỏ avgpool và fc layers nếu cần spatial features
                modules = list(resnet.children())[:-2]
                self.cnn = nn.Sequential(*modules)
                # Thêm conv layer để điều chỉnh số channels
                self.output_conv = nn.Conv2d(2048, output_dim, kernel_size=1)
            else:
                # Sử dụng toàn bộ ResNet nếu chỉ cần global features
                for param in resnet.parameters():
                    param.requires_grad = False
                in_features = resnet.fc.in_features
                resnet.fc = nn.Linear(in_features, output_dim)
                self.cnn = resnet
                self.output_conv = nn.Identity()

        else:
            # Xây dựng CNN từ đầu
            base_layers = [
                # Conv Block 1
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
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
            ]

            if use_spatial:
                # Chỉ thêm conv layer cuối nếu cần spatial features
                self.cnn = nn.Sequential(*base_layers)
                self.output_conv = nn.Conv2d(256, output_dim, kernel_size=1)
            else:
                # Thêm các layer fully connected nếu chỉ cần global features
                extra_layers = [
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, output_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                ]
                self.cnn = nn.Sequential(*(base_layers + extra_layers))
                self.output_conv = nn.Identity()

        self.use_spatial = use_spatial

    def forward(
        self, images: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Trích xuất đặc trưng từ ảnh
        Args:
            images: Batch ảnh [batch_size, 3, height, width]
        Returns:
            Nếu use_spatial=False:
                features: Global features [batch_size, output_dim]
            Nếu use_spatial=True:
                (spatial_features, global_features):
                - spatial_features: [batch_size, output_dim, h, w]
                - global_features: [batch_size, output_dim]
        """
        features = self.cnn(images)

        if self.use_spatial:
            features = self.output_conv(features)
            global_features = torch.mean(features, dim=[2, 3])
            return features, global_features
        else:
            return features
