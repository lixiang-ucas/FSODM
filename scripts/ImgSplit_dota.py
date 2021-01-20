import os
import data_utils as util
from Splitbase import splitbase


if __name__ == '__main__':
    import sys
    datapath = os.path.abspath(sys.argv[1])

    trainDir = os.path.join(datapath, 'trainset_reclabelTxt')
    evaDir = os.path.join(datapath, 'valset_reclabelTxt')
    train = os.listdir(trainDir)
    eva = os.listdir(evaDir)

    train = [x.strip('.txt') for x in train]
    eva = [x.strip('.txt') for x in eva]

    sets = ['training', 'evaluation']

    datalist = {'training': {}, 'evaluation': {}}
    for set in sets:
        if set == 'training':
            dlist = train
            dDir = trainDir
        else:
            dlist = eva
            dDir = evaDir

        for d in dlist:
            ipath = os.path.join(datapath, 'images/{}.png'.format(d))
            apath = os.path.join(dDir, '{}.txt'.format(d))

            datalist[set][d] = {}
            datalist[set][d]['imagepath'] = ipath
            datalist[set][d]['objects'] = util.parse_dota_poly(apath)

    # example usage of ImgSplit
    trainingsplit = splitbase(outdir=os.path.join(datapath, 'training'), ext='.png')
    trainingsplit.splitdata(datalist['training'], 1)

    evaluationsplit = splitbase(outdir=os.path.join(datapath, 'evaluation'), ext='.png')
    evaluationsplit.splitdata(datalist['evaluation'], 1)
