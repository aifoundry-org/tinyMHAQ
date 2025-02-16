from tinygrad import Tensor, nn

# def _weights_init(m):
#     classname = m.__class__.__name__
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal(m.weight)
        
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

def resnet20_cifar10(num_classes=10, pretrained=False):
    if not pretrained:
        return ResNet(BasicBlock, [3, 3, 3], num_classes)
    else: # TODO implement loading logic
        pass