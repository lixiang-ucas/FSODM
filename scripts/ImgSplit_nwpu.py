import os
import data_utils as util
import random
from Splitbase import splitbase


if __name__ == '__main__':
    import sys
    datapath = os.path.abspath(sys.argv[1])

    pfiles = os.listdir(os.path.join(datapath, 'NWPU/positive image set/'))
    nfiles = os.listdir(os.path.join(datapath, 'NWPU/negative image set/'))
    anno = os.listdir(os.path.join(datapath, 'NWPU/ground truth'))

    ptrain = random.sample(pfiles, len(pfiles) // 3 * 2)
    peva = ['p' + x.strip('.jpg') for x in pfiles if x not in ptrain]
    ptrain = ['p' + x.strip('.jpg') for x in ptrain]

    ntrain = random.sample(nfiles, len(nfiles) // 3 * 2)
    neva = ['n' + x.strip('.jpg') for x in nfiles if x not in ntrain]
    ntrain = ['n' + x.strip('.jpg') for x in ntrain]

    train = []
    eva = []

    train.extend(ptrain)
    train.extend(ntrain)
    eva.extend(peva)
    eva.extend(neva)

    sets = ['training', 'evaluation']
    i = 0
    outfilename = '{:0>3d}'

    datalist = {'training': {}, 'evaluation': {}}
    for set in sets:
        if set == 'training':
            dlist = train
        else:
            dlist = eva

        while len(dlist) != 0:
            file = random.sample(dlist, 1)[0]
            dlist.remove(file)
            if file.startswith('p'):
                ipath = os.path.join(datapath, 'NWPU/positive image set/{}.jpg'.format(file.strip('p')))
                apath = os.path.join(datapath, 'NWPU/ground truth/{}.txt'.format(file.strip('p')))
            else:
                ipath = os.path.join(datapath, 'NWPU/negative image set/{}.jpg'.format(file.strip('n')))
                apath = ''

            datalist[set][outfilename.format(i)] = {}
            datalist[set][outfilename.format(i)]['imagepath'] = ipath
            datalist[set][outfilename.format(i)]['objects'] = util.parse_nwpu_poly(apath)
            i += 1

    # example usage of ImgSplit
    trainingsplit = splitbase(outdir=os.path.join(datapath, 'training'), ext='.jpg')
    trainingsplit.splitdata(datalist['training'], 1)

    evaluationsplit = splitbase(outdir=os.path.join(datapath, 'evaluation'), ext='.jpg')
    evaluationsplit.splitdata(datalist['evaluation'], 1)