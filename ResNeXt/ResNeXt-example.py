import torchvision.models as models


model = models.resnext50_32x4d()

'''
kwargs['groups'] = 32
kwargs['width_per_group'] = 4
将上面两个参数传入到ResNet类中，通过这两个参数，计算bottleneck中的卷积核个数，如下代码所示base_width = width_per_group


width = int(planes * (base_width / 64.)) * groups
Both self.conv2 and self.downsample layers downsample the input when stride != 1
self.conv1 = conv1x1(inplanes, width)
self.bn1 = norm_layer(width)
self.conv2 = conv3x3(width, width, stride, groups, dilation)
'''