import os

if __name__ == '__main__':
    import sys

    datapath = os.path.abspath(sys.argv[1])
    datasetname = sys.argv[2]

    if datasetname == 'nwpu':
        classes = ['airplane',
                   'ship',
                   'storage-tank',
                   'baseball-diamond',
                   'tennis-court',
                   'basketball-court',
                   'ground-track-field',
                   'harbor',
                   'bridge',
                   'vehicle']
    elif datasetname == 'dior':
        classes = ['airplane',
                   'airport',
                   'baseballfield',
                   'basketballcourt',
                   'bridge',
                   'chimney',
                   'dam',
                   'Expressway-Service-area',
                   'Expressway-toll-station',
                   'golffield',
                   'groundtrackfield',
                   'harbor',
                   'overpass',
                   'ship',
                   'stadium',
                   'storagetank',
                   'tenniscourt',
                   'trainstation',
                   'vehicle',
                   'windmill']
    else:
        raise RuntimeError('Wrong dataset')

    fewshotlist = set()
    filelistDir = os.path.join(datapath, '{}list/'.format(datasetname))
    filelist = os.listdir(filelistDir)
    for file in filelist:
        if file.startswith('box_'):
            shots = file.split('_')[1].strip('shot')
            fewshotlist.add(shots)
    fewshotlist = list(fewshotlist)

    fulldict = 'data/{}_traindict_full.txt'.format(datasetname)
    with open(fulldict, 'w') as f:
        for cls in classes:
            line = '{} {}\n'.format(cls, os.path.join(filelistDir, '{}_training.txt'.format(cls)))
            print('{}:{}'.format(fulldict, line))
            f.write(line)

    for shot in fewshotlist:
        fewdict = 'data/{}_traindict_bbox_{}shot.txt'.format(datasetname, shot)
        with open(fewdict, 'w') as f:
            for cls in classes:
                line = '{} {}\n'.format(cls, os.path.join(filelistDir, 'box_{}shot_{}_train.txt'.format(shot, cls)))
                print('{}:{}'.format(fewdict, line))
                f.write(line)