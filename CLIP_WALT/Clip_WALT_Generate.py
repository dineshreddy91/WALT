#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:15:11 2022

@author: dinesh
"""

from collections import OrderedDict
from matplotlib import pyplot as plt
from utils import *
import scipy.interpolate

from scipy import interpolate
from clustering_utils import *
import glob
import cv2
from PIL import Image


import json
from psycopg2.extras import RealDictCursor
import psycopg2
import cv2


def ignore_indexes(tracks_all, labels_all):
    # get repeating bounding boxes
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    ignore_ind = []
    for index, track in enumerate(tracks_all):
        print('in ignore', index, len(tracks_all))
        if index in ignore_ind:
            continue

        if labels_all[index] < 1 or labels_all[index] > 3:
            ignore_ind.extend([index])            
        
        ind = get_indexes(track, tracks_all)
        if len(ind) > 30:
            ignore_ind.extend(ind)

    return ignore_ind
    
def repeated_indexes_old(tracks_all,ignore_ind, unoccluded_indexes=None):
    # get repeating bounding boxes
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if bb_intersection_over_union(x, y) > 0.8 and i not in ignore_ind]
    repeat_ind = []
    repeat_inds =[]
    if unoccluded_indexes == None:
        for index, track in enumerate(tracks_all):
            if index in repeat_ind or index in ignore_ind:
                continue
            ind = get_indexes(track, tracks_all)
            if len(ind) > 20:
                repeat_ind.extend(ind)
                repeat_inds.append([ind,track])
    else:
        for index in unoccluded_indexes:
            if index in repeat_ind or index in ignore_ind:
                continue
            ind = get_indexes(tracks_all[index], tracks_all)
            if len(ind) > 3:
                repeat_ind.extend(ind)
                repeat_inds.append([ind,tracks_all[index]])
    return repeat_inds

def get_unoccluded_instances(timestamps_final, tracks_all, ignore_ind, threshold = 0.01):
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x==y]
    unoccluded_indexes = []
    time_checked = []
    stationary_obj = []
    count =0 
    for time in np.unique(timestamps_final):
        print('In Unoccluded ', count, len(np.unique(timestamps_final)))
        count += 1
        if [time.year,time.month, time.day, time.hour, time.minute, time.second, time.microsecond] in time_checked:
            analyze_bb = []
            for ind in unoccluded_indexes_time:
                for ind_compare in  same_time_instances:
                    iou = bb_intersection_over_union(tracks_all[ind], tracks_all[ind_compare])
                    if  iou < 0.5 and iou > 0:
                        analyze_bb.extend([ind_compare])
                    if iou > 0.99:
                        stationary_obj.extend([str(ind_compare)+'+'+str(ind)])
                        
            for ind in  analyze_bb:
                occ = False
                for ind_compare in same_time_instances:
                    if bb_intersection_over_union_unoccluded(tracks_all[ind], tracks_all[ind_compare], threshold=threshold) > threshold and ind_compare != ind:
                        occ = True
                        break
                if occ == False:
                    unoccluded_indexes.extend([ind])
            continue
        
        same_time_instances = get_indexes(time,timestamps_final)
        unoccluded_indexes_time = []

        for ind in same_time_instances:
            if tracks_all[ind][4] < 0.9 or ind in ignore_ind:# or ind != 1859:
                continue
            occ = False
            for ind_compare in same_time_instances:
                if bb_intersection_over_union_unoccluded(tracks_all[ind], tracks_all[ind_compare], threshold=threshold) > threshold and ind_compare != ind and tracks_all[ind_compare][4] < 0.5:
                    occ = True
                    break
            if occ==False:
                unoccluded_indexes.extend([ind])
                unoccluded_indexes_time.extend([ind])
        time_checked.append([time.year,time.month, time.day, time.hour, time.minute, time.second, time.microsecond])
    return unoccluded_indexes,stationary_obj
                                
def visualize_unoccluded_detection(timestamps_final,tracks_all,segmentation_all, ignore_ind, unoccluded_indexes):            
    tracks_final = []
    tracks_final.append([])
    try:
        os.mkdir(camera_name + '/' + camera_name+'_unoccluded_car_detection/')
    except:
        print(camera_name + '/' + camera_name+'_unoccluded_car_detection/')
            
    for time in np.unique(timestamps_final):
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x==y]
        ind = get_indexes(time, timestamps_final)
        image_unocc = False
        for index in ind:
            if index not in unoccluded_indexes:
                continue
            else:
                image_unocc = True
                break
        if image_unocc == False:
            continue
            
        try:
            print('/media/data2/processedframes/'+camera_name+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/'+ str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg')
            image = np.array(Image.open('/media/data2/processedframes/'+camera_name+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/'+ str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg'))
        except:
            image_names = glob.glob('/home/dinesh/research/SWIN_OCCU/data/parking_real'+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/'+ str(time).replace(' ','T').split('.')[0]+'*')
            if len(image_names):
                image = np.array(Image.open(image_names[0]))
            else:
                print('image not found') 
                continue
        mask = image*0
        image_original = image.copy()
            
        for index in ind:
            track = tracks_all[index]

            if index in ignore_ind:
                continue
            if index not in unoccluded_indexes:
                continue
            try:
                bb_left, bb_top, bb_width, bb_height, confidence, id = track
            except:
                bb_left, bb_top, bb_width, bb_height, confidence = track

            if confidence > 0.6:
                mask = poly_seg(image, segmentation_all[index])
        cv2.imwrite(camera_name + '/' + camera_name+'_unoccluded_car_detection/' + str(index)+'.png', mask[:, :, ::-1])

def repeated_indexes(tracks_all,ignore_ind, unoccluded_indexes=None):
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if bb_intersection_over_union(x, y) > 0.8 and i not in ignore_ind]
    repeat_ind = []
    repeat_inds =[]
    if unoccluded_indexes == None:
        for index, track in enumerate(tracks_all):
            if index in repeat_ind or index in ignore_ind:
                continue

            ind = get_indexes(track, tracks_all)
            if len(ind) > 10:
                repeat_ind.extend(ind)
                repeat_inds.append([ind,track])
    else:
        for index in unoccluded_indexes:
            if index in repeat_ind or index in ignore_ind:
                continue
            ind = get_indexes(tracks_all[index], tracks_all)
            if len(ind) > 10:
                repeat_ind.extend(ind)
                repeat_inds.append([ind,tracks_all[index]])
        

    return repeat_inds
def poly_seg(image, segm):
    poly = np.array(segm).reshape((int(len(segm)/2), 2))
    overlay = image.copy()
    alpha = 0.5

    cv2.fillPoly(overlay, [poly], color=(255, 255, 0))
    cv2.addWeighted(overlay, 0.5, image, 1 - alpha, 0, image)
    return image

def visualize_unoccuded_clusters(repeat_inds, tracks, segmentation_all, timestamps_final):
    for index_, repeat_ind in enumerate(repeat_inds):
        try:
            image = np.array(Image.open('/media/data2/processedframes/'+camera_name+'/'+'T18-median_image.jpg'))
        except:
            time = timestamps_final[0]
            image = np.array(Image.open('/media/data2/processedframes/'+camera_name+'/'+str(time.year)+'-'+str(time.month) +'/' + str(timestamps_final[0]).replace(' ','T').replace(':','-').split('+')[0] + '.jpg'))
            print('mean image not computed')
        try:        
            os.mkdir(camera_name+ '/cropped/')
        except:
            print('folder exists')
        try:
            os.mkdir(camera_name+ '/cropped/' + str(index_) +'/')
        except:
            print(camera_name+ '/cropped/' + str(index_) +'/')
        for i in repeat_ind[0]:
            try:
                bb_left, bb_top, bb_width, bb_height, confidence = tracks[i]#bbox
            except:
                bb_left, bb_top, bb_width, bb_height, confidence, track_id = tracks[i]#bbox
                    
            cv2.rectangle(image,(int(bb_left), int(bb_top)),(int(bb_left+bb_width), int(bb_top+bb_height)),(0, 0, 255), 2)
            time = timestamps_final[i]
            
            try:
                image1 = np.array(Image.open('/media/data2/processedframes/'+camera_name+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/'+ str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg'))
            except:
                image_names = glob.glob('/home/dinesh/research/SWIN_OCCU/data/parking_real'+'/'+str(time.year)+'-'+str(time.month) +'/' + str(timestamps_final[i]).replace(' ','T').split('.')[0]+'*')
                if len(image_names):
                    image1 = np.array(Image.open(image_names[0]))
                else:
                    print('image not found') 
                    continue
            crop = image1[int(bb_top): int(bb_top + bb_height), int(bb_left):int(bb_left + bb_width)]
            print(camera_name+ '/cropped/' + str(index_) +'/o_' + str(i) +'.jpg')
            cv2.imwrite(camera_name+ '/cropped/' + str(index_) +'/o_' + str(i) +'.jpg', crop[:, :, ::-1])
            print(crop.shape)
            image1 = poly_seg(image1,segmentation_all[i])
            crop = image1[int(bb_top): int(bb_top + bb_height), int(bb_left):int(bb_left + bb_width)]
            cv2.imwrite(camera_name+ '/cropped/' + str(index_) +'/' + str(i)+'.jpg', crop[:, :, ::-1])
        if index_ > 100:
            break

        cv2.imwrite(camera_name+ '/cropped/' +  str(index_) +'.jpg', image[:, :, ::-1])
        
if __name__ == '__main__':
    camera_name = 'fifth_craig3'
    json_file_path = camera_name + '/' + camera_name + '.json'
    try:
        os.mkdir(camera_name)
    except:
        print(camera_name)
    print('ignore indices')
    unoccluded_indexes, stationary = get_unoccluded_instances(timestamps_final, tracks_all, ignore_ind, threshold = 0.05)
    
    tracks_all_unoccluded = [tracks_all[i] for i in unoccluded_indexes]
    segmentation_all_unoccluded = [segmentation_all[i] for i in unoccluded_indexes]
    labels_all_unoccluded = [labels_all[i] for i in unoccluded_indexes]
    timestamps_final_unoccluded = [timestamps_final[i] for i in unoccluded_indexes]
    np.savez(json_file_path,tracks_all_unoccluded=tracks_all_unoccluded, segmentation_all_unoccluded=segmentation_all_unoccluded, labels_all_unoccluded=labels_all_unoccluded, timestamps_final_unoccluded=timestamps_final_unoccluded )

    repeat_inds_clusters = repeated_indexes(tracks_all_unoccluded,[])
    visualize_unoccuded_clusters(repeat_inds_clusters, tracks_all_unoccluded, segmentation_all_unoccluded, timestamps_final_unoccluded)
    np.savez(json_file_path + '_clubbed', repeat_inds=repeat_inds_clusters)
    np.savez(json_file_path + '_stationary', stationary=stationary)
