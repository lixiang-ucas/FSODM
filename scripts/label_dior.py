import os
from PIL import Image


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


sets = ['training', 'evaluation']

training_classes = ['airplane',
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


if __name__ == '__main__':
    import sys
    datapath = os.path.abspath(sys.argv[1])

    labelpath = os.path.join(datapath, 'labels')
    if not os.path.exists(labelpath):
        os.mkdir(labelpath)

    for set in sets:
        files = os.listdir(os.path.join(datapath,'{}/images'.format(set)))
        image_ids = [x.strip('.jpg').strip('.png') for x in files]
        list_file = os.path.join(datapath, '{}.txt'.format(set))

        with open(list_file, 'w') as out_f:
            for id in image_ids:
                in_file = os.path.join(datapath, '{}/annotations/{}.txt'.format(set, id))
                out_file = os.path.join(datapath, 'labels/{}.txt'.format(id))
                image = os.path.join(datapath, '{}/images/{}.jpg'.format(set, id))

                im = Image.open(image)
                width, height = im.size

                with open(in_file, 'r') as in_f:
                    objs = [x.strip().split(' ') for x in in_f.readlines()]

                write_text = []
                print(in_file)
                for obj in objs:
                    if len(obj) < 5:
                        continue
                    cls = obj[4]

                    cls_id = training_classes.index(cls)
                    b = (float(obj[0]), float(obj[2]), float(obj[1]), float(obj[3]))
                    bb = convert((width, height), b)
                    write_text.append(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

                with open(out_file, 'w') as f:
                    for txt in write_text:
                        f.write(txt)
                out_f.write('{}/{}/images/{}.jpg\n'.format(datapath, set, id))
