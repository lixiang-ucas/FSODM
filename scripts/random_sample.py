# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
import sys
from os import path

sys.path.append(path.split(sys.path[0])[0])

import xml.etree.ElementTree as ET
import os, sys
import cPickle
import numpy as np
import argparse
from termcolor import colored
from utils import *
import pdb
import cv2
import random

cfg = {}


def get_novels(root, id=None):
    if root.endswith('txt'):
        if id == 'None':
            return []
        with open(root, 'r') as f:
            novels = f.readlines()
        return novels[int(id)].strip().split(',')
    else:
        return root.split(',')


def filter(detlines, clsfile):
    # pdb.set_trace()
    with open(clsfile, 'r') as f:
        imgids = [l.split()[0] for l in f.readlines() if l.split()[1] == '1']
    dls = [dl for dl in detlines if dl[0] in imgids]

    # dls = [dl for dl in dls if float(dl[1]) > 0.05]
    return dls


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    objects = []

    if cfg['data'] == 'nwpu':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': classes[int(obj[4]) - 1],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    elif cfg['data'] == 'dior':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': obj[4],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    else:
        raise RuntimeError('No dataset issued')

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetpath,
             classname,
             novel_class,
             best_num,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images

    files = os.listdir(imagesetpath)
    imagenames = [x.strip('.png').strip('.jpg') for x in files]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))

        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox, 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # pdb.set_trace()
    clsfile = path.join(path.dirname(imagesetpath), '{}_test.txt')
    clsfile = clsfile.format(classname)
    splitlines = [x.strip().split(' ') for x in lines]
    # print('before', len(splitlines))
    if args.single:
        print('before', len(splitlines))
        splitlines = filter(splitlines, clsfile)
        print('after', len(splitlines))
    # splitlines = bbox_filter(splitlines, conf=0.02)
    # print('after', len(splitlines))

    image_dict = {}
    for line in splitlines:
        if line[0] not in image_dict:
            image_dict[line[0]] = {}
        if 'confidence' not in image_dict[line[0]]:
            image_dict[line[0]]['confidence'] = []
        image_dict[line[0]]['confidence'].append(float(line[1]))
        if 'bbox' not in image_dict[line[0]]:
            image_dict[line[0]]['bbox'] = np.array([float(z) for z in line[2:]])[np.newaxis, :]
        else:
            image_dict[line[0]]['bbox'] = np.vstack((image_dict[line[0]]['bbox'],
                                                     np.array([float(z) for z in line[2:]])))

    result = {}
    for image in class_recs:
        if image not in image_dict or class_recs[image]['bbox'].size == 0:
            continue

        R = class_recs[image]
        BB = image_dict[image]['bbox'].copy()
        conf = np.array(image_dict[image]['confidence'])
        BBGT = R['bbox'].copy()
        tp = []
        fp = []
        fn = []

        sorted_ind = np.argsort(-conf)
        BB = BB[sorted_ind, :] if len(BB) != 0 else BB

        for d, bb in enumerate(BB):
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp.append(bb)
                    R['det'][jmax] = True
                else:
                    fp.append(bb)
            else:
                fp.append(bb)

        for idet, det in enumerate(R['det']):
            if not det:
                fn.append(R['bbox'][idet])

        result[image] = {'tp': tp, 'fp': fp, 'fn': fn}

    randoms = random.sample(result.keys(), best_num)
    file_path = path.join(imagesetpath, '{}.jpg')
    n = 0
    for r in randoms:
        random_samp = result[r]
        img = cv2.imread(file_path.format(r))

        for bbox in random_samp['tp']:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        for bbox in random_samp['fp']:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

        for bbox in random_samp['fn']:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        n += 1
        cv2.imwrite('./{}_example{}.jpg'.format(classname, n + 1), img)

    return randoms


def _do_python_eval(res_prefix, conf_path, best_num, novel=False, output_dir='output'):
    global cfg
    cfg = read_data_cfg(conf_path)
    # _devkit_path = '/data2/bykang/pytorch-yolo2/VOCdevkit'
    _devkit_path = os.path.split(cfg['valid'])[0]
    _year = '2007'
    dataset_name = cfg['data']
    _novel_file = cfg['novel']
    novelid = cfg['novelid']
    print('novelid: {}'.format(novelid))
    _novel_classes = get_novels(_novel_file, novelid)

    filename = res_prefix + '{:s}.txt'
    annopath = os.path.join(_devkit_path, 'evaluation', 'annotations', '{:s}.txt')
    imagesetpath = os.path.join(_devkit_path, 'evaluation', 'images')
    cachedir = os.path.join(_devkit_path, 'annotations_cache')
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(_year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(_novel_classes):
        random = voc_eval(filename,
                          annopath,
                          imagesetpath,
                          cls,
                          _novel_classes,
                          best_num,
                          cachedir,
                          ovthresh=0.5,
                          use_07_metric=use_07_metric)

        msg = '{}: '.format(cls)
        for ib, b in enumerate(random):
            msg += 'sample{}-{} '.format(ib + 1, b)
        print(msg)


if __name__ == '__main__':
    # res_prefix = '/data/hongji/darknet/project/voc/results/comp4_det_test_'
    parser = argparse.ArgumentParser()
    parser.add_argument('res_prefix', type=str)
    parser.add_argument('conf_path', type=str)
    parser.add_argument('best_num', type=int)
    parser.add_argument('--novel', action='store_true')
    parser.add_argument('--single', action='store_true')
    args = parser.parse_args()
    args.novel = True
    print('prefix: {}'.format(args.res_prefix))
    print('config file path: {}'.format(args.conf_path))
    _do_python_eval(args.res_prefix, args.conf_path, args.best_num, novel=args.novel, output_dir='output')
