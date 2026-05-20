import torch.nn as nn


class DeepPlainCNN(nn.Module):
    def __init__(self, num_classes=100, use_leaky=False):
        super().__init__()

        activation = nn.LeakyReLU(0.1) if use_leaky else nn.ReLU()

        layers = []
        # Initial expansion
        layers.extend([nn.Conv2d(3, 64, kernel_size=3, padding=1), activation])

        # 30 layers of constant channel width (NO BATCH NORM HERE!)
        for _ in range(30):
            layers.extend([
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                activation
            ])

        layers.append(nn.MaxPool2d(4, 4))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))