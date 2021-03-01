import os
import cv2
import sys
import math
import scipy
import random
import rasterio
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import seed
from glob import glob
from rasterio.windows import Window

from keras.models import model_from_json
from keras import backend as K
from keras.layers import Conv2D
from keras import layers
from keras.models import Model
import tensorflow as tf
import random
import json
import copy

from augmentation import augmentation 

class Generator:
    def __init__(self, batch_size, class_0, class_1, num_channels):
        self.num_channels = num_channels
        self.num_classes = 1
        self.IMG_ROW = 128
        self.IMG_COL = 128
        self.batch_size = batch_size
        self.class_0 = class_0
        self.class_1 = class_1
        self.cloud = False
        self.augm = False
        self.color_aug_prob = 0
        self.stands_id = False
        self.per_stand_loss = False
        self.instance_augm = False
        self.instance_augm_prob = 0.6
        self.background_prob = 0.6
        self.shadows = False
        self.extra_objects = False
        
        self.prob_split = False
        
        self.channels = ['pre_r', 'pre_g', 'pre_b']
        self.channels_background = ['RED', 'GRN', 'BLU']
        self.val = False
        self.background_list_val = []
        self.background_list_train = []
        
        self.val_upper_threshold = 0
        self.val_lower_threshold = 1620
        self.train_upper_threshold = 1620
        self.train_lower_threshold = 4418
            
    def get_img_mask_array(self, imgpath, upper_left_x, upper_left_y, pol_width, pol_height, crop_id, age_flag = False):
        with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
            size_x = src.width
            size_y = src.height
        difference_x = max(0, self.IMG_COL - int(pol_width))
        difference_y = max(0, self.IMG_ROW - int(pol_height))
        
        rnd_x = random.randint(max(0, int(upper_left_x) - difference_x),
                               min(size_x, int(upper_left_x) + int(pol_width) + difference_x) - self.IMG_COL)
        rnd_y = random.randint(max(0, int(upper_left_y) - difference_y),
                               min(size_y, int(upper_left_y) + int(pol_height) + difference_y) - self.IMG_ROW)
        
        window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
        
        mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
        for cl_name in self.class_0:
            #if '{}.tif'.format(cl_name) in os.listdir(imgpath):
            with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                mask_0 += src.read(window=window).astype(np.uint8)
                
        if self.stands_id:
            with rasterio.open(imgpath + '/id_mask.tif') as src:
                mask_id = src.read(window=window).astype(np.uint16)
                #tiff.imshow(mask_id)
                mask_id = np.where(mask_id[0,:,:]==int(crop_id), 1, 0) 
        #tiff.imshow(mask_id)
        #tiff.imshow(mask_0)
        img = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.uint8)
        for i, ch in enumerate(self.channels):
            with rasterio.open(imgpath+'/'+ch+ '.tif') as src:
                img[:,:,i] = src.read(window=window)
        
        #img_initial = copy.deepcopy(img)
        mask = np.ones((self.IMG_ROW, self.IMG_COL, self.num_classes))
        
        mask[:,:,0] = mask_0 
        
        flag_instance_augm = False
        
        if self.instance_augm and random.random() < self.instance_augm_prob: #self.val==False
            
            #---------------------------------------------------------------------------------------
            # crop target object
            #---------------------------------------------------------------------------------------           
            mask[:,:,0] = mask_0 * mask_id
            
            mask_tmp= np.ones((self.IMG_ROW, self.IMG_ROW, 3))
            mask_tmp[:,:,0] = mask[:, :, 0]>0.5
            mask_tmp[:,:,1] = mask[:, :, 0]>0.5
            mask_tmp[:,:,2] = mask[:, :, 0]>0.5
            
            # crop builbing
            building_mask = np.ones((self.IMG_ROW, self.IMG_ROW, 3))
            building_mask[:,:,0] = img[:, :,0] * mask_id
            building_mask[:,:,1] = img[:, :,1] * mask_id
            building_mask[:,:,2] = img[:, :,2] * mask_id
            
            #angle = 0# random.randint(0, 180)
            #new_img = rotate(building_mask, angle)
            
            #rows, colomns, _ = np.where(new_img>0)
            
            #rnd_x = 0 # random.randint(max(0, rows[0] - (128-rows[-1]+rows[0])),
                               #min(new_img.shape[0], rows[0] + 128) - 128)

            #rnd_y = 0 #random.randint(max(0, colomns[0] - (128-colomns[-1]+colomns[0])),
                              # min(new_img.shape[1], colomns[0] + 128) - 128)
            #img = new_img[rnd_x:rnd_x+self.IMG_ROW,rnd_y:rnd_y+self.IMG_ROW,:]
            
            #mask[:,:,0] = rotate(mask_tmp.astype(np.uint8), angle)[rnd_x:rnd_x+self.IMG_ROW,rnd_y:rnd_y+self.IMG_ROW,0]
            
            # add extra objects
            if self.extra_objects:
                for _ in range(random.randint(0, self.extra_objects)):
                    random_key = random.choice(list(self.json_file_cl0_train.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_train[random_key])

                    imgpath = random_key[:-len(random_key.split('_')[-1])-1]
                    crop_id = random_key.split('_')[-1]
                    #print('imgpath ', imgpath)
                    #print('img_name ', img_name)
                    with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
                        size_x = src.width
                        size_y = src.height
                        
                    difference_x = max(0, self.IMG_COL - int(pol_width))
                    difference_y = max(0, self.IMG_ROW - int(pol_height))
                    
                    intersect_initial = True
                    search_iter = 0
                    while intersect_initial and search_iter < 10:
                        #print(search_iter)
                        search_iter += 1
                        #rnd_x = random.randint(max(0, int(upper_left_x) - difference_x),
                        #                       min(size_x, int(upper_left_x) + int(pol_width) + difference_x) - self.IMG_COL)
                        #rnd_y = random.randint(max(0, int(upper_left_y) - difference_y),
                        #                       min(size_y, int(upper_left_y) + int(pol_height) + difference_y) - self.IMG_ROW)
                        
                        rnd_x = random.randint(0, size_x - self.IMG_COL)
                        rnd_y = random.randint(0, size_y - self.IMG_ROW)


                        window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)

                        mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
                        for cl_name in self.class_0:
                            with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                                mask_0 += src.read(window=window).astype(np.uint8)

                        with rasterio.open(imgpath + '/id_mask.tif') as src:
                            mask_id = src.read(window=window).astype(np.uint16)
                            mask_id = np.where(mask_id[0,:,:]==int(crop_id), 1, 0) 
                        
                        if np.sum(mask_id * mask[:,:,0]) == 0:
                            #print('miu')
                            mask[:,:,0] += mask_id
                            intersect_initial = False
                            img_extra = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.uint8)
                            for i, ch in enumerate(self.channels):
                                with rasterio.open(imgpath+'/'+ch+ '.tif') as src:
                                    img_extra[:,:,i] = src.read(window=window)
                            building_mask[:,:,0] = img_extra[:, :,0] * mask_id + building_mask[:,:,0] * np.where(mask_id, 0, 1)
                            building_mask[:,:,1] = img_extra[:, :,1] * mask_id + building_mask[:,:,1] * np.where(mask_id, 0, 1)
                            building_mask[:,:,2] = img_extra[:, :,2] * mask_id + building_mask[:,:,2] * np.where(mask_id, 0, 1)

            # add augm for object crop
            if self.augm:
                building_mask, mask_tmp = augmentation(building_mask.astype(np.uint8), mask, self.color_aug_prob)
                if len(mask_tmp.shape)==2:
                    mask[:,:,0]=mask_tmp
                else:
                    mask=mask_tmp
            building_mask = building_mask.astype(np.uint8)
            #---------------------------------------------------------------------------------------
            # find background
            #---------------------------------------------------------------------------------------
            flag_background = False
            if len(self.background_list_train) and random.random() < self.background_prob:
                if self.val:
                    imgpath = random.choice(self.background_list_val)
                else:
                    imgpath = random.choice(self.background_list_train) 
                flag_background = True
            else:
                attempt = 0
                if self.val:
                    random_key = random.choice(list(self.json_file_cl0_val.keys()))
                else:
                    random_key = random.choice(list(self.json_file_cl0_train.keys()))
                imgpath = random_key[:-len(random_key.split('_')[-1])-1]

                with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
                    size_x = src.width
                    size_y = src.height

                rnd_x = random.randint(0, size_x-self.IMG_ROW-1)
                rnd_y = random.randint(0, size_y-self.IMG_ROW-1)
                window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
                mask_background = np.zeros((1, self.IMG_ROW, self.IMG_COL))
                with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                    mask_background += src.read(window=window).astype(np.uint8)

                while np.sum(mask_background)>0:
                    attempt += 1
                    rnd_x = random.randint(0, size_x-self.IMG_ROW-1)
                    rnd_y = random.randint(0, size_y-self.IMG_ROW-1)
                    window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
                    mask_background = np.zeros((1, self.IMG_ROW, self.IMG_COL))
                    with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                        mask_background += src.read(window=window).astype(np.uint8)

                    if attempt > 50:
                        attempt = 0
                        if self.val:
                            random_key = random.choice(list(self.json_file_cl0_val.keys()))
                        else:
                            random_key = random.choice(list(self.json_file_cl0_train.keys()))
                        imgpath = random_key[:-len(random_key.split('_')[-1])-1]
                        with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
                            size_x = src.width
                            size_y = src.height


            background = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.uint8)
            if flag_background:
                channels_list = self.channels_background
                with rasterio.open(imgpath+'/'+ self.channels_background[0] + '.tif') as src:
                    size_x = src.width
                    size_y = src.height

                rnd_x = random.randint(0, size_x-self.IMG_ROW-1)
                rnd_y = random.randint(0, size_y-self.IMG_ROW-1)
                window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
            else:
                channels_list = self.channels
            for i, ch in enumerate(channels_list):
                with rasterio.open(imgpath+'/'+ch+ '.tif') as src:
                    background[:,:,i] = src.read(window=window)
            
            # add augm for background
            if self.augm:
                background, mask_tmp  = augmentation(background, mask, self.color_aug_prob)
                background = background.astype(np.uint8)
            
            
            #img = ((background*np.where(building_mask[:, :, 0]==0, 1, 0)) + building_mask).astype(np.uint8)
            mask_tmp = np.zeros((self.IMG_ROW, self.IMG_COL, self.num_channels))
            mask_tmp[:,:,0] = mask[:, :, 0]==0
            mask_tmp[:,:,1] = mask[:, :, 0]==0
            mask_tmp[:,:,2] = mask[:, :, 0]==0
            img = ((background*mask_tmp) + building_mask).astype(np.uint8)
            
            # add shadows
            if self.shadows:
                mask_shift = mask[:, :, 0]*1
        
                shift_x = random.randint(0,6) 
                shift_y = 4
                for i_sh in range(1, shift_x):
                    mask_shift[:-i_sh, :-i_sh-4] += mask[i_sh:, i_sh+4: , 0]

                shadow = (mask_shift>0)* (mask[:, :, 0]==0)
                alpha = random.choice([0.4, .3, .45])
                
                for i in range(3):
                    img[:,:,i] = img[:,:,i]*(shadow==0)+(alpha*(shadow>0)*img[:,:,i])
                
            flag_instance_augm = True
            
        #---------------------------------------------------------------------------------------
        # base augmentation
        #---------------------------------------------------------------------------------------
        #if self.val:
        #    img, mask_tmp  = augmentation(img, mask, 1.)
        #    if len(mask_tmp.shape)==2:
        #        mask[:,:,0]=mask_tmp
        #    else:
        #        mask=mask_tmp
                
        if self.augm and flag_instance_augm==False:
            img, mask_tmp  = augmentation(img, mask, self.color_aug_prob)
            if len(mask_tmp.shape)==2:
                mask[:,:,0]=mask_tmp
            else:
                mask=mask_tmp
                
        img = img / 255.
        img = img.clip(0, 1)
        return np.asarray(img), np.asarray(mask) #, np.asarray(background/ 255.), np.asarray(img_initial/ 255.)
    
    def extract_val(self, sample):
        return sample['upper_left_x'], sample['upper_left_y'], sample['pol_width'], sample['pol_height']
    
    def train_gen(self):
        while(True):
            self.val = False
            #self.background_prob = 0.5
            imgarr=[]
            maskarr=[]
            for i in range(self.batch_size):
                random_key = random.choice(list(self.json_file_cl0_train.keys()))
                upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_train[random_key])
                
                img_name = random_key[:-len(random_key.split('_')[-1])-1]
                img,mask=self.get_img_mask_array(img_name, upper_left_x, upper_left_y, 
                                                 pol_width, pol_height, random_key.split('_')[-1])
                imgarr.append(img)
                maskarr.append(mask)
            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[] 

    def val_gen(self):
        while(True):
            self.val = True
            #self.background_prob = 0.
            imgarr=[]
            maskarr=[]
            #background = []
            #img_initial = []
            for i in range(self.batch_size):
                random_key = random.choice(list(self.json_file_cl0_val.keys()))
                upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_val[random_key])
                
                img_name = random_key[:-len(random_key.split('_')[-1])-1]
                img,mask =self.get_img_mask_array(img_name, upper_left_x, upper_left_y, 
                                                 pol_width, pol_height, random_key.split('_')[-1])
                #background.append(background_img)
                maskarr.append(mask)
                imgarr.append(img)
                #img_initial.append(initial)
            yield (np.asarray(imgarr),np.asarray(maskarr)) #, np.asarray(background), np.asarray(img_initial))
            imgarr=[]
            maskarr=[]
   
    def read_json(self, folders, class_name):
        js_full = {}
        samples_set = set()
        for folder in folders:
            json_file = '{}/{}.json'.format(folder, 'all')
            with open(json_file, 'r') as f:
                js_tmp = json.load(f)
            keys_list = set(js_tmp.keys())
            for key in keys_list:
                js_tmp[folder+'_'+key] = js_tmp[key]
                del js_tmp[key]
            js_full.update(js_tmp)
        return js_full    
    
    def train_val_split_prob(self, js_full, split_ration):                
        seed(1)
        train_samples, val_samples = {}, {}
        keys_list = set(js_full.keys())
        for key in keys_list:
            if random.random() < split_ration:
                train_samples[key] = js_full[key]
            else:
                val_samples[key] = js_full[key]
            del js_full[key]

        return train_samples, val_samples
    
    '''
    def train_val_split(self, js_full, split_ration):               
        seed(1)
        train_samples, val_samples = {}, {}
        keys_list = set(js_full.keys())
        for key in keys_list:
            if js_full[key]["upper_left_y"] > 1620: # this threshold is for Venture image
                train_samples[key] = js_full[key]
            else:
                val_samples[key] = js_full[key]
            del js_full[key]

        return train_samples, val_samples
    '''
    def train_val_split(self, js_full, split_ration):               
        seed(1)
        train_samples, val_samples = {}, {}
        keys_list = set(js_full.keys())
        for key in keys_list:
            if js_full[key]["upper_left_y"] > self.train_upper_threshold and js_full[key]["upper_left_y"] < self.train_lower_threshold: # this threshold is for Venture image
                train_samples[key] = js_full[key]
            elif js_full[key]["upper_left_y"] < self.val_lower_threshold and js_full[key]["upper_left_y"] > self.val_upper_threshold:
                val_samples[key] = js_full[key]
            del js_full[key]
        return train_samples, val_samples
        #self.val_upper_threshold = 0
        #self.val_lower_threshold = 1620
        #self.train_upper_threshold = 1620
        #self.train_lower_threshold = 4418
          

    
    def load_dataset(self, folders, json_name_cl0, json_name_cl1, folders_val = None, split_ration=0.7):
        self.json_file_cl0_train = self.read_json(folders, json_name_cl0)
        
        if folders_val != None:
            self.json_file_cl0_val = self.read_json(folders_val, json_name_cl0)
        elif self.prob_split:
            self.json_file_cl0_train, self.json_file_cl0_val = self.train_val_split_prob(self.read_json(folders, json_name_cl0), split_ration)
        else:
            self.json_file_cl0_train, self.json_file_cl0_val = self.train_val_split(self.read_json(folders, json_name_cl0), split_ration)
            
def f1_score(pred, mask):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    TP += np.sum(pred[:,:,0]*mask[0,:,:])
    FN += np.sum((pred[:,:,0]==0)*mask[0,:,:])

    TN += np.sum((pred[:,:,0]==0)*(mask[0,:,:]==0))
    FP += np.sum((pred[:,:,0]==1)*(mask[0,:,:]==0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN) 
    f1_cl = 2*((precision*recall)/(precision+recall))

    return round(f1_cl, 3) 

def rotate(img, angle):
    (height, width) = img.shape[:2]
    (cent_x, cent_y) = (width // 2, height // 2)

    mat = cv2.getRotationMatrix2D((cent_x, cent_y), -angle, 1.0)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])

    n_width = int((height * sin) + (width * cos))
    n_height = int((height * cos) + (width * sin))

    mat[0, 2] += (n_width / 2) - cent_x
    mat[1, 2] += (n_height / 2) - cent_y

    return cv2.warpAffine(img, mat, (n_width, n_height))

