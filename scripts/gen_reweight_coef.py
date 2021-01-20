from darknet_meta_yolov3_spp import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from cfg import cfg
from cfg import parse_cfg
import os
import pdb

classes = ['airplane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
           'ground-track-field', 'harbor', 'bridge', 'vehicle']


def valid(datacfg, darknetcfg, learnetcfg, weightfile, outfile, use_baserw=False):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    metadict = options['meta']
    # name_list = options['names']
    # backup = cfg.backup
    ckpt = weightfile.split('/')[-1].split('.')[0]
    backup = weightfile.split('/')[-2]
    ckpt_pre = '/ene_' if use_baserw else '/ene'
    prefix = 'results/' + backup.split('/')[-1] + ckpt_pre + ckpt
    print('saving to: ' + prefix)
    # prefix = 'results/' + weightfile.split('/')[1]
    # names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    m = Darknet(darknetcfg, learnetcfg)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()

    kwargs = {'num_workers': 4, 'pin_memory': True}

    metaset = dataset.MetaDataset(metafiles=metadict, train=False, ensemble=True, with_ids=True)
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=64,
        shuffle=False,
        **kwargs
    )
    # metaloader = iter(metaloader)
    n_cls = len(metaset.classes)

    coef = [[[] for j in range(n_cls)] for i in range(3)]
    cnt = [0.0] * n_cls
    print('===> Generating dynamic weights...')
    kkk = 0
    for metax, mask, clsids in metaloader:
        print('===> {}/{}'.format(kkk, len(metaset) // 64))
        kkk += 1
        metax, mask = metax.cuda(), mask.cuda()
        metax, mask = Variable(metax, volatile=True), Variable(mask, volatile=True)
        dws = m.meta_forward(metax, mask)
        for ci, c in enumerate(clsids):
            for i in range(3):
                coef[i][c].append(dws[i][ci])

    outfile = './reweight_coef.data'
    with open(outfile, 'w') as f:
        for c in range(n_cls):
            print('processing %s' % classes[c])
            f.write(classes[c] + '\n')

            for i in range(3):
                f.write('coef%d\n' % i)
                for dw in coef[i][c]:
                    for n in dw:
                        f.write('%e ' % n.data[0])
                    f.write('\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) in [5, 6, 7]:
        datacfg = sys.argv[1]
        darknet = parse_cfg(sys.argv[2])
        learnet = parse_cfg(sys.argv[3])
        weightfile = sys.argv[4]
        if len(sys.argv) >= 6:
            gpu = sys.argv[5]
        else:
            gpu = '0'
        if len(sys.argv) == 7:
            use_baserw = True
        else:
            use_baserw = False

        data_options  = read_data_cfg(datacfg)
        net_options   = darknet[0]
        meta_options  = learnet[0]
        data_options['gpus'] = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # Configure options
        cfg.config_data(data_options)
        cfg.config_meta(meta_options)
        cfg.config_net(net_options)

        outfile = 'comp4_det_test_'
        valid(datacfg, darknet, learnet, weightfile, outfile, use_baserw)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
