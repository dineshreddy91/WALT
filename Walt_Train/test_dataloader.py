import argparse

import mmcv
import torch
from mmcv import Config, DictAction

from code_local.datasets import (build_dataloader, build_dataset)
from mmdet.core import visualization as vis
import numpy as np
import cv2
from matplotlib.patches import Polygon
import pycocotools.mask as mask_util


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')

    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args

def show_result(img,
                result,
                labels,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):

        img = img.copy()

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        print(bbox_result)
        bboxes = np.vstack(bbox_result)
        bboxes = np.hstack([bboxes,np.ones((len(labels),1))])
        print(bboxes, segm_result)
        #labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        segms.dtype='uint8'
        segms = segms*120
        cv2.imwrite('image.png',segms)

        print(segms.shape)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        segms = None
        CLASSES = str(np.zeros((1000,1)))
        # draw bounding boxes
        img = vis.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        return img
        #if not (show or out_file):
        #   return img

def visualize_2d_3dbboxes_on_image(img, bb_2d_all, bb_3d_proj_all):
    lines = [[0,1],[1,3],[0,2],[3,2],[0,4],[1,5],[2,6],[3,7],[4,5],[5,7],[4,6],[7,6]]
    img_bb = img.copy()
    for bbox in bb_2d_all:
        if bbox[0] < 0.00000001 or bbox[2] < 0.00000001:
            continue
        try:
            cv2.rectangle(img_bb,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0),1)
        except:
            continue
    for points2d_all in bb_3d_proj_all:
        for point in points2d_all:
            if point[0] <0.000000001 or point[1] < 0.000001:
                continue
            try:
                cv2.circle(img, (int(point[0]),int(point[1])),radius=3, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
            except:
                continue
        for line in lines:
            if points2d_all[line[0]][0] < 0.00000001 or points2d_all[line[0]][1] < 0.00000001 or points2d_all[line[1]][0] < 0.000001 or points2d_all[line[0]][1] < 0.00000001:
                continue
            try:
                cv2.line(img, (int(points2d_all[line[0]][0]),int(points2d_all[line[0]][1])), (int(points2d_all[line[1]][0]),int(points2d_all[line[1]][1])), color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
            except:
                continue

    cv2.imwrite('test.png', img_bb)
    cv2.imwrite('test2.png', img)



def visualize_3d(bb_3d_all):
    return
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True

    print(cfg.data.val)
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    cfg.data.samples_per_gpu = 1 


    dataset = [build_dataset(cfg.data.train)]
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]
    for i, data_batch in enumerate(data_loaders):
        #print(data_batch.__dict__)
        for j, data in enumerate(data_batch):
            print(data['img_metas'])
            images = data['img']
            try:
                img = images.data[0].numpy()
            except:
                img = images[0].numpy()

            img = np.transpose(img[0], (1, 2, 0))
            img[:,:,2] = ((img[:,:,0] - img[:,:,0].min())/img[:,:,0].max())*255
            img[:,:,1] = ((img[:,:,1] - img[:,:,1].min())/img[:,:,1].max())*255
            img[:,:,0] = ((img[:,:,2] - img[:,:,2].min())/img[:,:,2].max())*255
            img = img.astype(int)

            gt_bboxes = data['gt_bboxes']
            gt_masks = data['gt_masks']
            #gt_bboxes_3d = data['gt_bboxes_3d']
            #gt_bboxes_3d_proj = data['gt_bboxes_3d_proj']
            gt_labels = data['gt_labels']
            img_new = img.copy()
            #gt_masks.data[0][0].masks[0,1,1] = 2
            print(np.unique(gt_masks.data[0][0].masks))
            img = show_result(img, (gt_bboxes.data[0][0].numpy(), gt_masks.data[0][0]), gt_labels.data[0][0].numpy())
            cv2.imwrite('test3.png', img)
            print(gt_masks.data[0][0])
            #if len(gt_masks.data[0][0]) >1:
            #    break
            break

            #asas
            #img_original = cv2.imread(data['img_metas'].data[0][0]['filename'])
            #cv2.imwrite('test4.png', img_original)
            #visualize_2d_3dbboxes_on_image(img_original, gt_bboxes.data[0][0].numpy(), gt_bboxes_3d_proj.data[0][0].numpy())
            #visualize_2d_3dbboxes_on_image(img, gt_bboxes.data[0][0].numpy(), gt_bboxes_3d_proj.data[0][0].numpy())


if __name__ == '__main__':
    main()
