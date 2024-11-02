import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import torchvision.models as models


# 下面的模块是根据所指定的模型筛选出指定层的特征图输出，
# 如果未指定也就是extracted_layers是None则以字典的形式输出全部的特征图，
# 另外因为全连接层本身是一维的没必要输出因此进行了过滤。

class FeatureExtractor(nn.Module):
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
            # print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = img[6: 400, 83: 477]
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(pic_dir):

    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)
    img = img.to(device)

    # 这里主要是一些参数，比如要提取的网络，网络的权重，要提取的层，指定的图像放大的大小，存储路径等等。
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
