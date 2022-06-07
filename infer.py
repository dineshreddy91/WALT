from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.mask.utils import encode_mask_results
import numpy as np
import mmcv
import torch 
from imantics import Polygons, Mask
import json

class detections():
    def __init__(self, cfg_path, device, model_path = 'detections_mm/models/mask_rcnn_swin_tiny_patch4_window7.pth'):
        self.model = init_detector(cfg_path, model_path, device=device)
        self.all_preds = []
        self.all_scores = []
        self.index = []
        self.score_thr = 0.92
        self.result = []
        self.record_dict = {'model': cfg_path,'results': []}
        self.detect_count = []


    def run_on_image(self, image):
        self.result = inference_detector(self.model, image)
        image_labelled = self.model.show_result(image, self.result, score_thr=self.score_thr)
        return image_labelled

    def process_output(self, count):
        result = self.result
        infer_result = {'url': count,
                        'boxes': [],
                        'scores': [],
                        'keypoints': [],
                        'segmentation': [],
                        'label_ids': [],
                        'track': [],
                        'labels': []}

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            #segm_result = encode_mask_results(segm_result)
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
            #else:
            #    bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        #labels = [np.full(bbox.shape[0], i) for i, bbox in enumerate(bbox_result)]

        labels = np.concatenate(labels)
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        for i, (bbox, label, segm) in enumerate(zip(bboxes, labels, segms)):
            if bbox[-1].item() <0.3:
                continue
            box = [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]
            polygons = Mask(segm).polygons()
            
            infer_result['boxes'].append(box)
            infer_result['segmentation'].append(polygons.segmentation)
            infer_result['scores'].append(bbox[-1].item())
            infer_result['labels'].append(self.model.CLASSES[label])
            infer_result['label_ids'].append(label)
        self.record_dict['results'].append(infer_result)
        self.detect_count = labels
        #self.all_preds.append(self.final_results)
        #self.all_scores.append(self.scores)
        #self.index.append(count)
    
    def write_json(self, filename):
        with open(filename + '.json', 'w') as f:
            json.dump(self.record_dict, f)
        '''
            file = open(filename + '.pkl', 'wb')
        pickle.dump(self.all_preds , file)
        pickle.dump(self.all_scores , file)
        pickle.dump(self.index , file)
        file.close()
        '''



def main():
    #detect_people = detections('configs_local/swin/occ_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_people_real_coco.py', 'cuda:0', model_path='work_dirs/occ_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_people_real_coco/latest.pth')
    detect = detections('Walt_Train/configs_local/swin/occ_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_parking_real_coco.py', 'cuda:0', model_path='data/pretrained/occ_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_parking_real_coco/latest.pth')
    #detect = detections('configs_local/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_kins.py', 'cuda:0', model_path='work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_kins/latest.pth')
    #configs_local/swin/occ_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_parking_real_coco.py','cpu', model_path='work_dirs/occ_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_parking_real_coco/latest.pth')
    import cv2, glob
    filenames = sorted(glob.glob('/code/demo/paper_results/*'))
    filenames = sorted(glob.glob('/code/demo/seq1/*'))
    count = 0
    for filename in filenames:
        #img=cv2.imread('data/parking_real_test/iii1_gen/images/2021-10-25T01-02-15.233384.jpg')
        img=cv2.imread(filename)
        try:
            #img = detect_people.run_on_image(img)
            img = detect.run_on_image(img)
        except:
            continue
        count=count+1
        import os

        try: 
            import os
            os.makedirs(os.path.dirname(filename.replace('demo','demo/results_ours/')))
            os.mkdirs(os.path.dirname(filename))
        except:
            print('done')
        #cv2.imwrite(filename.replace('demo','demo/results_ours/')+'_seg.png',img)
        cv2.imwrite(filename.replace('demo','demo/results_ours/') +'combined.png',img)
        cv2.imwrite('check.jpg',img)
        if count == 30000:
            break
        try:
            detect.process_output(count)
        except:
            continue

    print(detect.record_dict)
    np.savez('FC', a= detect.record_dict)
    with open('check.json', 'w') as f:
        json.dump(detect.record_dict, f)
    detect.write_json('seq3')
    asas
    detect.process_output(0)
main()
