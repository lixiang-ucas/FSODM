import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLossV2
from cfg import *
# from dynamic_conv import DynamicConv2d
from dynamic_conv import dynamic_conv2d
from pooling import GlobalMaxPool2d
from pooling import GlobalAvgPool2d
from pooling import Split
import pdb


# from layers.batchnorm.bn import BN2d


def maybe_repeat(x1, x2):
    n1 = x1.size(0)
    n2 = x2.size(0)
    if n1 == n2:
        pass
    elif n1 < n2:
        assert n2 % n1 == 0
        shape = x1.shape[1:]
        nc = n2 // n1
        x1 = x1.repeat(nc, *[1] * x1.dim())
        x1 = x1.transpose(0, 1).contiguous()
        x1 = x1.view(-1, *shape)
    else:
        assert n1 % n2 == 0
        shape = x2.shape[1:]
        nc = n1 // n2
        x2 = x2.repeat(nc, *[1] * x2.dim())
        x2 = x2.transpose(0, 1).contiguous()
        x2 = x2.view(-1, *shape)
    return x1, x2


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, darknet_file, learnet_file):
        super(Darknet, self).__init__()
        self.blocks = darknet_file if isinstance(darknet_file, list) else parse_cfg(darknet_file)
        self.learnet_blocks = learnet_file if isinstance(learnet_file, list) else parse_cfg(learnet_file)
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        self.models, self.routs = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.learnet_models, self.learnet_routs = self.create_network(self.learnet_blocks)
        self.loss = self.models[len(self.models) - 1]

        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def meta_forward(self, metax, mask):
        # Get weights from learnet
        done_split = False
        for i in range(int(self.learnet_blocks[0]['feat_layer'])):
            if i == 0 and metax.size(1) == 6:
                done_split = True
                metax = torch.cat(torch.split(metax, 3, dim=1))
            metax = self.models[i](metax)
        if done_split:
            metax = torch.cat(torch.split(metax, int(metax.size(0) / 2)), dim=1)
        if cfg.metain_type in [2, 3]:
            metax = torch.cat([metax, mask], dim=1)

        dynamic_weights = []
        ind = -2
        layer_outputs = []
        for block in self.learnet_blocks:
            ind += 1

            if block['type'] == 'learnet':
                continue
            elif block['type'] in ['convolutional', 'maxpool']:
                metax = self.learnet_models[ind](metax)
            elif block['type'] == 'globalmax':
                metax = self.learnet_models[ind](metax)
                dynamic_weights.append(metax)
            elif block['type'] == 'route':
                layers = [int(x) for x in block['layers'].split(',')]
                if len(layers) == 1:
                    metax = layer_outputs[layers[0]]
                else:
                    try:
                        metax = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        metax = torch.cat([layer_outputs[i] for i in layers], 1)
            layer_outputs.append(metax if ind in self.learnet_routs else [])

        return dynamic_weights

    def detect_forward(self, x, dynamic_weights):
        # Perform detection
        ind = -2
        dynamic_cnt = 2
        layer_outputs = []
        output = []

        for block in self.blocks:
            ind = ind + 1
            # if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'upsample']:
                if self.is_dynamic(block):
                    x = self.models[ind]((x, dynamic_weights[dynamic_cnt]))
                    dynamic_cnt -= 1
                else:
                    x = self.models[ind](x)

                if 'output_layer' in block and int(block['output_layer']) == 1:
                    output.append(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)))
            elif block['type'] == 'route':
                layers = [int(x) for x in block['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif block['type'] == 'shortcut':
                x = x + layer_outputs[int(block['from'])]
            elif block['type'] == 'region':
                continue
            else:
                print('unknown type %s' % (block['type']))
            layer_outputs.append(x if ind in self.routs else [])

        return torch.cat(output, 2)

    def forward(self, x, metax, mask, ids=None):
        # pdb.set_trace()
        dynamic_weights = self.meta_forward(metax, mask)
        x = self.detect_forward(x, dynamic_weights)
        return x

    def print_network(self):
        print_cfg(self.blocks)
        print('---------------------------------------------------------------------')
        print_cfg(self.learnet_blocks)

    def create_network(self, blocks):
        hyperparams = blocks[0]
        output_filters = [int(hyperparams['channels'])]
        module_list = nn.ModuleList()
        routs = []  # list of layers which rout to deeper layers
        ind = -2
        filters = -1

        for mdef in blocks:
            ind += 1
            modules = nn.Sequential()

            if mdef['type'] in ['net', 'learnet']:
                continue
            if mdef['type'] == 'convolutional':
                bn = int(mdef['batch_normalize'])
                filters = int(mdef['filters'])
                size = int(mdef['size'])
                stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
                pad = (size - 1) // 2 if int(mdef['pad']) else 0
                dynamic = True if 'dynamic' in mdef and int(mdef['dynamic']) == 1 else False

                if dynamic:
                    partial = int(mdef['partial']) if 'partial' in mdef else None
                    Conv2d = dynamic_conv2d(is_first=True, partial=partial)
                else:
                    Conv2d = nn.Conv2d

                modules.add_module('Conv2d', Conv2d(in_channels=output_filters[-1],
                                                    out_channels=filters,
                                                    kernel_size=size,
                                                    stride=stride,
                                                    padding=pad,
                                                    groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                    bias=not bn))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
                if mdef['activation'] == 'leaky':
                    # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                    # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                elif mdef['activation'] == 'swish':
                    modules.add_module('activation', Swish())

            elif mdef['type'] == 'maxpool':
                size = int(mdef['size'])
                stride = int(mdef['stride'])
                maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))
                if size == 2 and stride == 1:  # yolov3-tiny
                    modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                    modules.add_module('MaxPool2d', maxpool)
                else:
                    modules = maxpool

            elif mdef['type'] == 'upsample':
                modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

            elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
                layers = [int(x) for x in mdef['layers'].split(',')]
                filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
                routs.extend([l if l > 0 else l + ind for l in layers])
                modules = EmptyModule()
                # if mdef[i+1]['type'] == 'reorg3d':
                #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

            elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
                filters = output_filters[int(mdef['from'])]
                layer = int(mdef['from'])
                routs.extend([ind + layer if layer < 0 else layer])
                modules = EmptyModule()

            elif mdef['type'] == 'region':
                loss = RegionLossV2()
                anchors = mdef['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(mdef['classes'])
                loss.num_anchors = int(mdef['num'])
                loss.object_scale = float(mdef['object_scale'])
                loss.noobject_scale = float(mdef['noobject_scale'])
                loss.class_scale = float(mdef['class_scale'])
                loss.coord_scale = float(mdef['coord_scale'])
                loss.input_size = (self.height, self.width)
                modules = loss

            elif mdef['type'] == 'globalmax':
                modules = GlobalMaxPool2d()

            elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
                # torch.Size([16, 128, 104, 104])
                # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
                pass

            else:
                print('Warning: Unrecognized Layer Type: ' + mdef['type'])

            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        return module_list, routs

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        for blocks, models in [(self.blocks, self.models), (self.learnet_blocks, self.learnet_models)]:
            ind = -2
            for block in blocks:
                if start >= buf.size:
                    break
                ind = ind + 1
                if block['type'] == 'net' or block['type'] == 'learnet':
                    continue
                elif block['type'] == 'convolutional':
                    model = models[ind]
                    if self.is_dynamic(block) and model[0].weight is None:
                        continue
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(buf, start, model[0], model[1])
                    else:

                        start = load_conv(buf, start, model[0])
                elif block['type'] == 'connected':
                    model = models[ind]
                    if block['activation'] != 'linear':
                        start = load_fc(buf, start, model[0])
                    else:
                        start = load_fc(buf, start, model)
                elif block['type'] == 'maxpool':
                    pass
                elif block['type'] == 'reorg':
                    pass
                elif block['type'] == 'route':
                    pass
                elif block['type'] == 'shortcut':
                    pass
                elif block['type'] == 'region':
                    pass
                elif block['type'] == 'avgpool':
                    pass
                elif block['type'] == 'softmax':
                    pass
                elif block['type'] == 'cost':
                    pass
                elif block['type'] == 'globalmax':
                    pass
                elif block['type'] == 'globalavg':
                    pass
                elif block['type'] == 'split':
                    pass
                else:
                    print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        # pdb.set_trace()
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1 + len(self.learnet_blocks)

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff + 1):
            # pdb.set_trace()
            if blockId >= len(self.blocks):
                if blockId == len(self.blocks):
                    ind = -2
                blockId = blockId - len(self.blocks)
                blocks = self.learnet_blocks
                models = self.learnet_models
            else:
                blocks = self.blocks
                models = self.models
            ind = ind + 1

            block = blocks[blockId]
            if block['type'] == 'convolutional':
                model = models[ind]
                if self.is_dynamic(block) and model[0].weight is None:
                    continue
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = models[ind]
                if block['activation'] == 'linear':
                    save_fc(fp, model)
                else:
                    save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'globalmax':
                pass
            elif block['type'] == 'learnet':
                pass
            elif block['type'] == 'globalavg':
                pass
            elif block['type'] == 'split':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

    def is_dynamic(self, block):
        return 'dynamic' in block and int(block['dynamic']) == 1


if __name__ == '__main__':
    import argparse
    from torch.autograd import Variable

    parser = argparse.ArgumentParser()
    parser.add_argument('--darknet', type=str, required=True)
    parser.add_argument('--learnet', type=str, required=True)
    args = parser.parse_args()

    net = Darknet(args.darknet, args.learnet)
    net = net.cuda()

    x = Variable(torch.randn(8, 3, 416, 416))
    metax = Variable(torch.randn(8, 3, 384, 384))
    mask = Variable(torch.randn(8, 1, 96, 96))
    x = x.cuda()
    metax = metax.cuda()
    mask = mask.cuda()

    y = net(x, metax, mask)
    pdb.set_trace()
    net.save_weights('/tmp/dynamic.weights')
    print('hello')


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))
