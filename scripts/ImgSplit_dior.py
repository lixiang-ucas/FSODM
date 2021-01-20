import os
import data_utils as util
from Splitbase import splitbase


if __name__ == '__main__':
    import sys
    datapath = os.path.abspath(sys.argv[1])

    trainlistfile = os.path.join(datapath, 'Main/train.txt')
    evalistfile = os.path.join(datapath, 'Main/val.txt')
    testlistfile = os.path.join(datapath, 'Main/test.txt')
    with open(trainlistfile, 'r') as f:
        train = f.readlines()
    with open(evalistfile, 'r') as f:
        train.extend(f.readlines())
    with open(testlistfile, 'r') as f:
        eva = f.readlines()

    train = [x.strip() for x in train]
    eva = [x.strip() for x in eva]

    sets = ['training', 'evaluation']

    datalist = {'training': {}, 'evaluation': {}}
    for set in sets:
        if set == 'training':
            dlist = train
            imageFile = os.path.join(datapath, 'JPEGImages-trainval/{}.jpg')
        else:
            dlist = eva
            imageFile = os.path.join(datapath, 'JPEGImages-test/{}.jpg')

        for d in dlist:
            ipath = imageFile.format(d)
            apath = os.path.join(datapath, 'Annotations/{}.xml'.format(d))

            datalist[set][d] = {}
            datalist[set][d]['imagepath'] = ipath
            datalist[set][d]['objects'] = util.parse_dior_poly(apath)

    # example usage of ImgSplit
    trainingsplit = splitbase(outdir=os.path.join(datapath, 'training'), ext='.jpg')
    trainingsplit.splitdata(datalist['training'], 1)

    evaluationsplit = splitbase(outdir=os.path.join(datapath, 'evaluation'), ext='.jpg')
    evaluationsplit.splitdata(datalist['evaluation'], 1)
