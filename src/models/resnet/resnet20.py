from tinygrad import Tensor, nn
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch, get_child

# def _weights_init(m):
#     classname = m.__class__.__name__
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal(m.weight)

weights = {
    'resnet20_orig': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/refs/heads/master/pretrained_models/resnet20-12fca82f.th',
    'resnet20': 'https://github.com/aifoundry-org/MHAQ/raw/refs/heads/bad_temper/saved_models_pytorch/cifar10_ResNet20v1_best.th',
    # 'resnet32':'https://github.com/akamaster/pytorch_resnet_cifar10/raw/refs/heads/master/pretrained_models/resnet32-d509ac18.th',
    # 'resnet44':'https://github.com/akamaster/pytorch_resnet_cifar10/raw/refs/heads/master/pretrained_models/resnet44-014dd654.th',
    # 'resnet56':'https://github.com/akamaster/pytorch_resnet_cifar10/raw/refs/heads/master/pretrained_models/resnet44-014dd654.th',
    # 'resnet110':'https://github.com/akamaster/pytorch_resnet_cifar10/raw/refs/heads/master/pretrained_models/resnet44-014dd654.th',
    # 'resnet1202':'https://github.com/akamaster/pytorch_resnet_cifar10/raw/refs/heads/master/pretrained_models/resnet1202-f3b1deed.th'
}


class LambdaLayer:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, x: Tensor) -> Tensor:
        return self.lambd(x)


class BasicBlock:
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = []
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = LambdaLayer(
                lambda x: x[:, :, ::2, ::2].pad(
                    (0, 0, 0, 0, planes//4, planes//4),
                    "constant", 0
                )
            )
    
    def __call__(self, x: Tensor):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        out = out + x.sequential(self.shortcut)
        out = out.relu()
        return out

class ResNet:
    def __init__(self, block, num_blocks, num_classes=10):
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
    
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return layers
    
    def __call__(self, x: Tensor) -> Tensor:
        out = self.bn1(self.conv1(x)).relu()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.avg_pool(out.size()[3])
        out = out.flatten(1)
        out = self.linear(out)
        return out
    
    def forward(self, x: Tensor) -> Tensor:
        return self.__call__(x)

def resnet20_cifar10(num_classes=10, pretrained=False):
    if not pretrained:
        return ResNet(BasicBlock, [3, 3, 3], num_classes)
    else:
        resnet_ = ResNet(BasicBlock, [3, 3, 3], num_classes)
        url = weights['resnet20']
        load_from_pretrained(model=resnet_, url=url)

def load_from_pretrained(model, url):
    for k, dat in torch_load(fetch(url)).items():
      try:
        obj: Tensor = get_child(model, k)
      except AttributeError as e:
        if 'fc.' in k and model.fc is None:
          continue

        raise e

      if 'fc.' in k and obj.shape != dat.shape:
        print("skipping fully connected layer")
        continue # Skip FC if transfer learning

      if 'bn' not in k and 'downsample' not in k: assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat.to(obj.device).reshape(obj.shape))