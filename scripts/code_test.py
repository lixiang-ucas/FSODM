import cv2
import torch
import numpy as np
import os

if __name__ == '__main__':
    # import sys
    # datapath = os.path.abspath(sys.argv[1])
    #
    # classes = set()
    # files = os.listdir(os.path.join(datapath, 'training/annotations'))
    # path = os.path.join(datapath, 'training/annotations/{}')
    #
    # for file in files:
    #     print('Parsing {}'.format(file))
    #     with open(path.format(file), 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.strip().split()
    #             classes.add(line[-1])
    #
    # classes = list(classes)
    # classes.sort()
    # print(classes)
    import sys
    path = sys.argv[1]

    dirs = os.listdir(path)
    for d in dirs:
        if d in ['airplane', 'baseball-diamond', 'tennis-court']:
            txts = os.listdir(os.path.join(path, d))
            num = 0
            for t in txts:
                with open(os.path.join(path, d, t), 'r') as f:
                    num += len(f.readlines())

            print('{}: {}'.format(d, num))
