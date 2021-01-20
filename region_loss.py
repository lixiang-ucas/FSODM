import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils import *
from cfg import cfg
from numbers import Number
from random import random, randint
import pdb


def neg_filter(pred_boxes, target, withids=False):
    assert pred_boxes.size(0) == target.size(0)
    if cfg.neg_ratio == 'full':
        inds = list(range(pred_boxes.size(0)))
    elif isinstance(cfg.neg_ratio, Number):
        flags = torch.sum(target, 1) != 0
        flags = flags.cpu().data.tolist()
        ratio = cfg.neg_ratio * sum(flags) * 1. / (len(flags) - sum(flags))
        if ratio >= 1:
            inds = list(range(pred_boxes.size(0)))
        else:
            flags = [0 if f == 0 and random() > ratio else 1 for f in flags]
            inds = np.argwhere(flags).squeeze()
            pred_boxes, target = pred_boxes[inds], target[inds]
    else:
        raise NotImplementedError('neg_ratio not recognized')
    if withids:
        return pred_boxes, target, inds
    else:
        return pred_boxes, target


def neg_filter_v2(pred_boxes, target, withids=False):
    assert pred_boxes.size(0) == target.size(0)
    if cfg.neg_ratio == 'full':
        inds = list(range(pred_boxes.size(0)))
    elif isinstance(cfg.neg_ratio, Number):
        flags = torch.sum(target, 1) != 0
        flags = flags.cpu().data.tolist()
        ratio = cfg.neg_ratio * sum(flags) * 1. / (len(flags) - sum(flags))
        if ratio >= 1:
            inds = list(range(pred_boxes.size(0)))
        else:
            flags = [0 if f == 0 and random() > ratio else 1 for f in flags]
            if sum(flags) == 0:
                flags[randint(0, len(flags) - 1)] = 1
            inds = np.nonzero(flags)[0]
            pred_boxes, target = pred_boxes[inds], target[inds]
    else:
        raise NotImplementedError('neg_ratio not recognized')
    if withids:
        return pred_boxes, target, inds
    else:
        return pred_boxes, target


def build_targets(pred_boxes, target, conf, anchors, num_anchors, feature_size, input_size, ignore_thresh):
    nB = target.size(0)
    nA = num_anchors
    # print('anchor_step: ', anchor_step)
    obj_mask = torch.cuda.ByteTensor(nB, nA, feature_size[0], feature_size[1]).fill_(0)
    noobj_mask = torch.cuda.ByteTensor(nB, nA, feature_size[0], feature_size[1]).fill_(1)
    tx = torch.zeros(nB, nA, feature_size[0], feature_size[1]).cuda()
    ty = torch.zeros(nB, nA, feature_size[0], feature_size[1]).cuda()
    tw = torch.zeros(nB, nA, feature_size[0], feature_size[1]).cuda()
    th = torch.zeros(nB, nA, feature_size[0], feature_size[1]).cuda()
    tcls = torch.zeros(nB, nA, feature_size[0], feature_size[1]).cuda()
    iou_scores = torch.zeros(nB, nA, feature_size[0], feature_size[1]).cuda()

    tboxes = target.view(-1, 5)
    nonzero_ind = tboxes[:, 3] > 0
    tboxes = tboxes[nonzero_ind.unsqueeze(1).repeat(1, 5)].view(-1, 5)
    ind_B = torch.linspace(0, nB - 1, nB).unsqueeze(1).repeat(1, 50).view(-1).long().cuda()
    ind_B = ind_B[nonzero_ind]
    gx = (tboxes[:, 1] * feature_size[1]).float()
    gy = (tboxes[:, 2] * feature_size[0]).float()
    gw = (tboxes[:, 3] * input_size[1]).float()
    gh = (tboxes[:, 4] * input_size[0]).float()
    aw = anchors[:, 0]
    ah = anchors[:, 1]
    nbox = tboxes.size(0)
    gt_box = torch.cat([torch.zeros(1, nbox).cuda(), torch.zeros(1, nbox).cuda(), gw.unsqueeze(0), gh.unsqueeze(0)], 0)
    anchor_box = torch.cat([torch.zeros(1, nA).cuda(), torch.zeros(1, nA).cuda(), aw.unsqueeze(0), ah.unsqueeze(0)], 0)
    ious = bbox_ious(gt_box.unsqueeze(2).repeat(1, 1, nA), anchor_box.unsqueeze(1).repeat(1, nbox, 1), x1y1x2y2=False)
    best_ious, best_a = ious.max(1)
    gj = gy.long()
    gi = gx.long()
    obj_mask[ind_B, best_a, gj, gi] = 1
    noobj_mask[ind_B, best_a, gj, gi] = 0

    for i, iou in enumerate(ious):
        if (iou > ignore_thresh).sum():
            noobj_mask[ind_B[i:i + 1], (iou > ignore_thresh).nonzero().squeeze(1), gj[i:i + 1], gi[i:i + 1]] = 0

    tx[ind_B, best_a, gj, gi] = gx - gx.floor()
    ty[ind_B, best_a, gj, gi] = gy - gy.floor()
    tw[ind_B, best_a, gj, gi] = torch.log(gw / anchors[best_a][:, 0])
    th[ind_B, best_a, gj, gi] = torch.log(gh / anchors[best_a][:, 1])
    tcls[ind_B, best_a, gj, gi] = tboxes[:, 0].float()
    tconf = obj_mask.float()
    pred_boxes = pred_boxes.contiguous().view(nB, nA, feature_size[0], feature_size[1], 4).cuda()
    conf = conf.contiguous().view(nB, nA, feature_size[0], feature_size[1]).data
    target_boxes = torch.cat([(tboxes[:, 1] * input_size[1]).float().unsqueeze(0),
                              (tboxes[:, 2] * input_size[0]).float().unsqueeze(0),
                              gw.unsqueeze(0),
                              gh.unsqueeze(0)], 0)

    iou_scores[ind_B, best_a, gj, gi] = bbox_ious(pred_boxes[ind_B, best_a, gj, gi].t(), target_boxes, x1y1x2y2=False)
    conf50 = (conf[ind_B, best_a, gj, gi] > 0.5).float()
    detected50 = (iou_scores[ind_B, best_a, gj, gi] > 0.5).float() * conf50
    detected75 = (iou_scores[ind_B, best_a, gj, gi] > 0.75).float() * conf50

    return nbox, iou_scores, obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls, detected50, detected75


class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) / num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        # import pdb; pdb.set_trace()
        # output : BxAs*(4+1+num_classes)*H*W

        # if target.dim() == 3:
        #     # target : B * n_cls * l
        #     l = target.size(-1)
        #     target = target.permute(1,0,2).contiguous().view(-1, l)
        if target.dim() == 3:
            target = target.view(-1, target.size(-1))
        bef = target.size(0)
        output, target = neg_filter(output, target)
        # print("{}/{}".format(target.size(0), bef))

        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB, nA, (5 + nC), nH, nW)
        x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        # [nB, nA, nC, nW, nH] | (bs, 5, 1, 13, 13)
        cls = output.index_select(2, Variable(torch.linspace(5, 5 + nC - 1, nC).long().cuda()))
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)

        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes,
                                                                                                    target.data,
                                                                                                    self.anchors, nA,
                                                                                                    nC, \
                                                                                                    nH, nW,
                                                                                                    self.noobject_scale,
                                                                                                    self.object_scale,
                                                                                                    self.thresh,
                                                                                                    self.seen)
        cls_mask = (cls_mask == 1)
        if cfg.metayolo:
            tcls.zero_()
        nProposals = int((conf > 0.25).float().sum().data[0])

        tx = Variable(tx.cuda())
        ty = Variable(ty.cuda())
        tw = Variable(tw.cuda())
        th = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls = Variable(tcls.view(-1)[cls_mask].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask = Variable(conf_mask.cuda().sqrt())
        cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())
        cls = cls[cls_mask].view(-1, nC)

        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask) / 2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
            self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0],
            loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss


class RegionLossV2(nn.Module):
    """
    Yolo region loss + Softmax classification across meta-inputs
    """

    def __init__(self, num_classes=0, anchors=[], num_anchors=1, input_size=(832, 832)):
        super(RegionLossV2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.coord_scale = 1
        self.class_scale = 1
        self.obj_scale = 1
        self.noobj_scale = 100
        self.thresh = 0.5
        self.seen = 0
        self.input_size = input_size
        self.feature_scale = [32, 16, 8]
        print('class_scale', self.class_scale)

    def forward(self, output, target):
        # output : (bs*cs, nA*(5+1), N)
        # target : (bs, cs, 50*5)
        # Get all classification prediction
        # pdb.set_trace()
        bs = target.size(0)
        cs = target.size(1)
        nA = self.num_anchors
        nC = self.num_classes
        N = output.data.size(2)
        # feature_size = [[26, 26], [52, 52], [104, 104]]
        cls = output.view(output.size(0), nA, (5 + nC), N)
        cls = cls.index_select(2, Variable(torch.linspace(5, 5 + nC - 1, nC).long().cuda())).squeeze()
        cls = cls.view(bs, cs, nA * N).transpose(1, 2).contiguous().view(bs * nA * N, cs)
        cls_conf = F.softmax(cls, 1)
        _, cls_max_ids = torch.max(cls_conf, 1)
        cls_max_ids = cls_max_ids.data
        pre_cls_mask = torch.zeros(bs * nA * N, cs).cuda()
        pre_cls_mask[torch.linspace(0, bs * nA * N - 1, bs * nA * N).long().cuda(), cls_max_ids] = 1
        pre_cls_mask = pre_cls_mask.view(bs, nA * N, cs).transpose(1, 2).contiguous().view(bs * cs, nA, N)

        # Rearrange target and perform filtering operation
        target = target.view(-1, target.size(-1))
        # bef = target.size(0)
        output, target, inds = neg_filter_v2(output, target, withids=True)
        counts, _ = np.histogram(inds, bins=bs, range=(0, bs * cs))
        # print("{}/{}".format(target.size(0), bef))
        pre_cls_mask = pre_cls_mask[inds]

        t0 = time.time()
        nB = output.data.size(0)

        output = output.view(nB, nA, (5 + nC), N)  # (nB, nA, (5+nC), N)
        x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).squeeze(2))  # (nB, nA, N)
        y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).squeeze(2))
        w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).squeeze(2)
        h = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).squeeze(2)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).squeeze(2))
        # [nB, nA, nC, nW, nH] | (bs, 5, 1, 13, 13)
        # cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        # cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB, nA, N)
        grid_x = []
        grid_y = []
        anchor_w = []
        anchor_h = []
        scale = []
        feature_size = []
        for fs in self.feature_scale:
            feature_h = self.input_size[0] / fs
            feature_w = self.input_size[1] / fs
            feature_size.append([feature_h, feature_w])
            grid_x.append(torch.linspace(0, feature_w - 1, feature_w).repeat(feature_h, 1) \
                          .repeat(nB * nA, 1, 1).view(nB, nA, feature_h * feature_w).cuda())
            grid_y.append(torch.linspace(0, feature_h - 1, feature_h).repeat(feature_w, 1).t() \
                          .repeat(nB * nA, 1, 1).view(nB, nA, feature_h * feature_w).cuda())
            scale.append((torch.ones(nB, nA, feature_h * feature_w) * fs).cuda())
        grid_x = torch.cat(grid_x, 2)  # (nB, nA, N)
        grid_y = torch.cat(grid_y, 2)
        scale = torch.cat(scale, 2)
        for i in range(3):
            aw = torch.Tensor(self.anchors[6 * i:6 * (i + 1)]).view(nA, -1) \
                .index_select(1, torch.LongTensor([0])).cuda()
            ah = torch.Tensor(self.anchors[6 * i:6 * (i + 1)]).view(nA, -1) \
                .index_select(1, torch.LongTensor([1])).cuda()
            anchor_w.append(aw.repeat(nB, feature_size[i][0] * feature_size[i][1]) \
                            .view(nB, nA, feature_size[i][0] * feature_size[i][1]))
            anchor_h.append(ah.repeat(nB, feature_size[i][0] * feature_size[i][1]) \
                            .view(nB, nA, feature_size[i][0] * feature_size[i][1]))
        anchor_w = torch.cat(anchor_w, 2)
        anchor_h = torch.cat(anchor_h, 2)
        pred_boxes[0] = (x.data + grid_x) * scale
        pred_boxes[1] = (y.data + grid_y) * scale
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.permute(1, 2, 3, 0).contiguous())  # (nB, nA, N, 4)
        t2 = time.time()
        nGT = 0
        iou_scores = []
        obj_mask = []
        noobj_mask = []
        tx = []
        ty = []
        tw = []
        th = []
        tconf = []
        tcls = []
        start_N = 0
        detected50 = torch.zeros(0)
        detected75 = torch.zeros(0)
        for imap in xrange(3):
            nGT, iou_scores_temp, obj_mask_temp, noobj_mask_temp, tx_temp, ty_temp, tw_temp, th_temp, tconf_temp, \
            tcls_temp, detected50_temp, detected75_temp = build_targets(
                pred_boxes[:, :, start_N:start_N + feature_size[imap][0] * feature_size[imap][1], :],
                target.data.cuda(),
                conf[:, :, start_N:start_N + feature_size[imap][0] * feature_size[imap][1]],
                torch.Tensor(self.anchors[6 * imap:6 * (imap + 1)]).view(nA, -1).cuda(),
                nA,
                feature_size[imap],
                self.input_size,
                self.thresh)
            if not len(detected50):
                detected50 = torch.zeros(nGT).cuda()
            if not len(detected75):
                detected75 = torch.zeros(nGT).cuda()
            detected50 += detected50_temp
            detected75 += detected75_temp
            start_N += feature_size[imap][0] * feature_size[imap][1]
            iou_scores.append(iou_scores_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            obj_mask.append(obj_mask_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            noobj_mask.append(noobj_mask_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            tx.append(tx_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            ty.append(ty_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            tw.append(tw_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            th.append(th_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            tconf.append(tconf_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))
            tcls.append(tcls_temp.view(nB, nA, feature_size[imap][0] * feature_size[imap][1]))

        iou_scores = torch.cat(iou_scores, 2)
        obj_mask = torch.cat(obj_mask, 2)
        noobj_mask = torch.cat(noobj_mask, 2)
        tx = torch.cat(tx, 2)
        ty = torch.cat(ty, 2)
        tw = torch.cat(tw, 2)
        th = torch.cat(th, 2)
        tconf = torch.cat(tconf, 2)
        tcls = torch.cat(tcls, 2)

        # Take care of class mask
        idx_start = 0
        cls_mask_list = []
        tcls_list = []
        for i in range(len(counts)):
            if counts[i] == 0:
                cur_mask = torch.zeros(nA, N).cuda()
                cur_tcls = torch.zeros(nA, N).cuda()
            else:
                cur_mask = torch.sum(obj_mask[idx_start:idx_start + counts[i]].float(), dim=0)
                cur_tcls = torch.sum(tcls[idx_start:idx_start + counts[i]], dim=0)
            cls_mask_list.append(cur_mask)
            tcls_list.append(cur_tcls)
            idx_start += counts[i]
        cls_mask = torch.stack(cls_mask_list)  # (bs, nA, N)
        tcls = torch.stack(tcls_list)

        cls_mask = (cls_mask == 1)
        conf50 = (conf > 0.5).float().data
        iou50 = (iou_scores > 0.5).float()
        detected_mask = conf50 * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        detected50 = (detected50 > 0).float()
        detected75 = (detected75 > 0).float()
        recall50 = detected50.sum() / (nGT + 1e-16)
        recall75 = detected75.sum() / (nGT + 1e-16)

        tx = Variable(tx)
        ty = Variable(ty)
        tw = Variable(tw)
        th = Variable(th)
        tconf = Variable(tconf)

        obj_mask = Variable(obj_mask)
        noobj_mask = Variable(noobj_mask)
        # cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,cs).cuda())
        cls = cls[Variable(cls_mask.view(-1, 1).repeat(1, cs))].view(-1, cs)
        cls_max_ids = cls_max_ids[cls_mask.view(-1)]
        tcls = Variable(tcls[cls_mask].long())
        cls_acc = float(torch.sum(cls_max_ids == tcls.data)) / (cls_max_ids.numel() + 1e-16)

        ClassificationLoss = nn.CrossEntropyLoss()
        MseLoss = nn.MSELoss()
        BceLoss = nn.BCELoss()

        t3 = time.time()

        loss_x = self.coord_scale * MseLoss(x[obj_mask], tx[obj_mask])
        loss_y = self.coord_scale * MseLoss(y[obj_mask], ty[obj_mask])
        loss_w = self.coord_scale * MseLoss(w[obj_mask], tw[obj_mask])
        loss_h = self.coord_scale * MseLoss(h[obj_mask], th[obj_mask])
        loss_conf_obj = BceLoss(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = BceLoss(conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        if len(cls):
            loss_cls = self.class_scale * ClassificationLoss(cls, tcls)
        else:
            loss_cls = Variable(torch.Tensor([0]).float().cuda())

        # # pdb.set_trace()
        # ids = [9,11,12,16]
        # new_cls, new_tcls = select_classes(cls, tcls, ids)
        # new_tcls = Variable(torch.from_numpy(new_tcls).long().cuda())
        # loss_cls_new = self.class_scale * nn.CrossEntropyLoss(size_average=False)(new_cls, new_tcls)
        # loss_cls_new *= 10
        # loss_cls += loss_cls_new

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print(
            '%d: nGT %d, precision %f, recall50 %f, recall75 %f, cls_acc %f, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % \
            (self.seen, nGT, precision, recall50, recall75, cls_acc, loss_x.data[0], loss_y.data[0], \
             loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        # print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, cls_new %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss_cls_new.data[0], loss.data[0]))
        return loss


def select_classes(pred, tgt, ids):
    # convert tgt to numpy
    tgt = tgt.cpu().data.numpy()
    new_tgt = [(tgt == d) * i for i, d in enumerate(ids)]
    new_tgt = np.max(np.stack(new_tgt), axis=0)
    idxes = np.argwhere(new_tgt > 0).squeeze()
    new_pred = pred[idxes]
    new_pred = new_pred[:, ids]
    new_tgt = new_tgt[idxes]
    return new_pred, new_tgt
