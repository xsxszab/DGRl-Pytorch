
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, 0, 0.1)
        torch.nn.init.constant_(m.bias, 0)


class BRN(nn.Module):

    def __init__(self, k=3, n=5):
        """Init BRN.

        :param k: The k in original paper.
        :param n: The n in original paper.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(k + 1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Conv2d(128, n*n, 3, padding=1)

        self.conv.apply(init_weights)
        init_weights(self.conv7)
        with torch.no_grad():
            self.conv7.bias.data[12] = 1
        self.n = n

    def forward(self, img, sal):
        """merge original image and saliency map from RLN output to get better saliency prediction.

        :param img: tensor [Batch_size, 3, 384, 384]
        :param sal: tensor [Batch_size, 1, 384, 384]
        :return: saliency map with 2 channels [Batch_size, 2, 384, 384]
        """
        img = img * 0.00392157  # normalization
        feature = torch.cat([img, sal], 1)
        feature = self.conv(feature)
        feature = self.conv7(feature)

        sal = nn.functional.unfold(sal, kernel_size=self.n, padding=int((self.n-1)/2))
        feature = feature.view((1, self.n * self.n, 384 * 384))
        sal = sal * feature
        sal = nn.functional.fold(sal, output_size=(384, 384), kernel_size=self.n, padding=int((self.n-1)/2))
        return sal


if __name__ == '__main__':
    model = BRN()
    img = Image.open('test.jpg', 'r')
    img = img.resize((384, 384))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.from_numpy(img).float()
    sal = torch.ones(1, 1, 384, 384)
    output = model.forward(img, sal)
