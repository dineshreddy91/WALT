#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:14:47 2021

@author: dinesh
"""
import glob
from utils import bb_intersection_over_union_unoccluded
import numpy as np
from PIL import Image
import datetime
import cv2
import os


def get_image(time, folder):
    print(folder+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/' + str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg')
    image = np.array(Image.open(folder+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/' + str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg'))
    if image is None:
        try:
            image_names = glob.glob( folder+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/' + str(time).replace(' ','T').split('.')[0]+'*')
            image = np.array(Image.open(image_names[0]))
        except:
            print('file not found')
    return image

def get_mask(segm, image):
    poly = np.array(segm).reshape((int(len(segm)/2), 2))
    mask = image.copy()*0
    cv2.fillConvexPoly(mask, poly, (255, 255, 255))
    return mask

def get_unoccluded(indices, tracks_all):
    unoccluded_indexes = []
    unoccluded_index_all =[]
    while 1:
        unoccluded_clusters = []
        len_unocc = len(unoccluded_indexes)
        for ind in indices:
            if ind in unoccluded_indexes:
                continue
            occ = False
            for ind_compare in indices:
                if ind_compare in unoccluded_indexes:
                    continue
                if bb_intersection_over_union_unoccluded(tracks_all[ind], tracks_all[ind_compare]) > 0.01 and ind_compare != ind:
                    occ = True
            if occ==False:
                unoccluded_indexes.extend([ind])
                unoccluded_clusters.extend([ind])
        if len(unoccluded_indexes) == len_unocc and len_unocc != 0:
            for ind in indices:
                if ind not in unoccluded_indexes:
                    unoccluded_indexes.extend([ind])
                    unoccluded_clusters.extend([ind])
            
        unoccluded_index_all.append(unoccluded_clusters)
        if len(unoccluded_indexes) > len(indices)-5:
            break
    return unoccluded_index_all

def primes(n): # simple sieve of multiples 
   odds = range(3, n+1, 2)
   sieve = set(sum([list(range(q*q, n+1, q+q)) for q in odds], []))
   return [2] + [p for p in odds if p not in sieve]

def save_image(image_read,i,camera_name,tracks,timestamps,segmentations,path):
        image = image_read.copy()
        indices = np.random.randint(len(tracks),size=30)
        prime_numbers = primes(1000)
        unoccluded_index_all = get_unoccluded(indices, tracks)
        
        mask_stacked = image*0
        mask_stacked_all =[]
        count = 0
        time = datetime.datetime.now()
        try:
            os.mkdir(camera_name + '_gen')
            os.mkdir(camera_name + '_gen/images')
            os.mkdir(camera_name + '_gen/Segmentation')
        except:
            print(camera_name)

        for l in indices:
                try:
                    image_crop = get_image(timestamps[l], path)
                except:
                    continue
                try:
                    bb_left, bb_top, bb_width, bb_height, confidence = tracks[l]
                except:
                    bb_left, bb_top, bb_width, bb_height, confidence, track_id = tracks[l]
                mask = get_mask(segmentations[l], image)
                image[mask > 0] = image_crop[mask > 0]
                mask[mask > 0] = 1
                for count, mask_inc in enumerate(mask_stacked_all):
                    mask_stacked_all[count][cv2.bitwise_and(mask, mask_inc) > 0] = 2
                mask_stacked_all.append(mask)
                mask_stacked += mask
                count = count+1
        
        print(camera_name + '_gen/images/'+str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg')
        cv2.imwrite(camera_name + '_gen/images/'+str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg', image[:, :, ::-1])
        cv2.imwrite(camera_name + '_gen/Segmentation/'+str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg', mask_stacked[:, :, ::-1]*30)
                
def generate(): # simple sieve of multiples 
    camera_name = 'fifth_craig3'
    json_file_path1 = '{}/{}.json'.format(camera_name,camera_name)# iii1/iii1_7_test.json' # './data.json'
    path = '/media/data2/processedframes/' + camera_name
    
    data = np.load(json_file_path1+'.npz', allow_pickle=True)
    tracks = data['tracks_all_unoccluded']
    segmentations = data['segmentation_all_unoccluded']
    timestamps = data['timestamps_final_unoccluded']
    
    try:
        image_read = np.array(Image.open('/media/data2/processedframes/'+camera_name+'/'+'T18-median_image.jpg'))
    except:
        time = timestamps[12]
        image_read = np.array(Image.open('/media/data2/processedframes/'+camera_name+'/'+str(time.year)+'-'+str(time.month).zfill(2) +'/' + str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg'))
        print('mean image not computed')

    image_read = np.array(Image.open('/media/data2/processedframes/'+camera_name+'/'+'T18-median_image.jpg'))
    for i in range(max(int(len(tracks)/5),3000)):
        save_image(image_read,i,camera_name,tracks,timestamps,segmentations, path)

if __name__ == '__main__':
    generate()
