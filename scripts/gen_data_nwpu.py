import os
import random
import shutil

pfiles = os.listdir('./positive image set/')
nfiles = os.listdir('./negative image set/')
anno = os.listdir('./ground truth')

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

for set in sets:
    out_anno_path = './{}/annotations/'.format(set)
    if not os.path.exists(out_anno_path):
        os.makedirs(out_anno_path)
    out_img_path = './{}/images/'.format(set)
    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)

    if set == 'training':
        list = train
    else:
        list = eva

    while len(list) != 0:
        file = random.sample(list, 1)[0]
        list.remove(file)
        if file.startswith('p'):
            sname = './positive_image_set/{}.jpg'.format(file.strip('p'))
            shutil.copyfile('./ground_truth/{}.txt'.format(file.strip('p')), './{}/annotations/{:0>3d}.txt'.format(set, i))
        else:
            sname = './negative_image_set/{}.jpg'.format(file.strip('n'))
            f = open('./{}/annotations/{:0>3d}.txt'.format(set, i), 'w')
            f.close()

        shutil.copyfile(sname, './{}/images/{:0>3d}.jpg'.format(set, i))
        i += 1







