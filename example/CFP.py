
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import torchvision.models as models
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


""" Cited from "Centralized Feature Pyramid for Object Detection" """
""" DOI: https://doi.org/10.1109/TIP.2023.3297408 """

# The following module filters the output of the feature maps of the specified layers according to the specified model.
# If unspecified, i.e. extracted_layers is None, then all feature maps are output as a dictionary.
# In addition, since the fully connected layer itself is one-dimensional,
# there is no need to output it, so it is filtered.

class FeatureExtractor(nn.Module):
    """ Multi-scale feature fusion feature extraction """
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            x = module(x)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x
        return outputs


def get_picture(pic_name, transform):
    """ Obtaining original CGR image data """
    img = skimage.io.imread(pic_name)
    # Due to the blank space at the edge of the CGR scatter plot, we only intercept the part with scatter in the image
    img = img[6: 400, 83: 477]
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def get_feature(pic_dir):
    """ Obtaining Multiscale Fusion Features for CGR Images """
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Insert dimension
    img = img.unsqueeze(0)
    img = img.to(device)
    # The main parameters here are the network to be extracted, the weights of the network, the layers to be extracted,
    # the size of the specified image enlargement, the storage path and so on.
    net = models.resnet101().to(device)
    net.load_state_dict(torch.load('pre-trained model path/resnet101-5d3b4d8f.pth'))  # 预训练好的模型
    exact_list = None
    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    return outs


if __name__ == '__main__':
    path = 'data/CGR_example/CGR_example_1.png'
    outputs = get_feature(path)
    print(outputs['fc'])
