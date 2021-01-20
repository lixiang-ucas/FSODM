import os
from darknet import Darknet
import dataset
from utils import *
from cfg import cfg
from cfg import parse_cfg
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE


def get_coefs(datacfg, darknetcfg, learnetcfg, weightfile):
    options = read_data_cfg(datacfg)
    metadict = options['meta']

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
                coef[i][c].append(dws[i][ci].data.squeeze().cpu().numpy())

    return coef


if __name__ == '__main__':
    import sys

    datacfg = sys.argv[1]
    darknet = parse_cfg(sys.argv[2])
    learnet = parse_cfg(sys.argv[3])
    weightfile = sys.argv[4]
    if len(sys.argv) >= 6:
        gpu = sys.argv[5]
    else:
        gpu = '0'

    data_options = read_data_cfg(datacfg)
    net_options = darknet[0]
    meta_options = learnet[0]
    data_options['gpus'] = gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Configure options
    cfg.config_data(data_options)
    cfg.config_meta(meta_options)
    cfg.config_net(net_options)

    coefs = get_coefs(datacfg, darknet, learnet, weightfile)
    for i, coef in enumerate(coefs):
        coef = np.array(coef)
        coef = np.reshape(coef, (-1, coef.shape[-1]))
        coef_embad = TSNE(n_jobs=4).fit_transform(coef)
        np.savetxt('coef_{}.csv'.format(i), coef_embad, delimiter=',', fmt='%s')
