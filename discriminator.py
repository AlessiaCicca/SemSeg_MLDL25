import torch.nn as nn

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=32):
        super(FCDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
