# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:27:01 2018

@author: jasonsli
"""

import cv2
import numpy as np
import os
import nibabel as nib
import pandas as pd
'''
lesion_seg_result_root = 'E://submission//result_all//all_lesion_AH_clahe'
#liver_seg_result_root = 'E://submission//dice_crop_lesion'

cate = dict()
for item in os.listdir(lesion_seg_result_root):
    vol = item.split('-')[0]
    if not vol in cate:
        cate[vol] = []
    cate[vol].append(item)

nii_root = 'E://NII//NII//Test'
out_nii_root = 'D://seg_nii3//'
for vol in range(70):
    nii = nib.load(os.path.join(nii_root, 'test-volume-' + str(vol) + '.nii'))
    nii_data = nii.get_data()
    nii_data = np.zeros(nii_data.shape)

    ROI_csv = 'E://submission//ROI.csv'
    obj = pd.read_csv(ROI_csv, index_col = 0, header = None)
    for path in cate[str(vol)]:
        out = np.zeros((512,512))
        sli_num = int(path.split('-')[1][:-4])
        lesion_mask = cv2.imread(os.path.join(lesion_seg_result_root, path))
#    liver_mask = cv2.imread(os.path.join(liver_seg_result_root, path))
   
        x_min,x_max,y_min,y_max = obj.ix[str(vol)+'-'+str(sli_num)+'.png']
    
        x,y = round((x_min+x_max)/2),round((y_min+y_max)/2)
    
        ori_shape = max(x_max-x_min, y_max-y_min)
        lesion_mask = cv2.resize(lesion_mask, (ori_shape,ori_shape))
    
        out[int(y-ori_shape/2):int(y+ori_shape/2),int(x-ori_shape/2):int(x+ori_shape/2)] = lesion_mask[:,:,0]
    
        nii_data[:,:,sli_num-1] = (out>0)*2
    
    nii_data = nii_data.astype(np.int16)

    ni_img = nib.Nifti1Image(nii_data, nii.affine, nii.header)
    nib.save(ni_img, out_nii_root+'lesion-seg-{}.nii'.format(str(vol)))

D = nib.load('C://Users//jasonsli//Desktop//0.nii').get_data()
'''

import os

###
import cv2
import numpy as np
import os
import nibabel as nib
import pandas as pd

lesion_seg_result_root = 'E://submission//AH-net//lesion_result'
liver_seg_result_root = 'E://submission//AH-net//liver_result'

targets = os.listdir('E://submission//result_all//seg_liver_png_crop_weight_clip')

cate = dict()
for item in os.listdir(liver_seg_result_root):
    vol = item.split('-')[0]
    if not vol in cate:
        cate[vol] = []
    cate[vol].append(item)
for vol in range(60,70):

    for path in cate[str(vol)]:
        out = np.zeros((512,512))
        sli_num = int(path.split('-')[1][:-4])
        lesion_mask = cv2.imread(os.path.join(lesion_seg_result_root, path))
        liver_mask = cv2.imread(os.path.join(liver_seg_result_root, path))
        if path not in targets:
            lesion_mask = np.zeros((512,512,3))
            liver_mask = np.zeros((512,512,3))

        cv2.imwrite('E://submission//AH-net//liver_result_hard//'+path, (liver_mask[:,:,0]>255*0.3)*255)
        cv2.imwrite('E://submission//AH-net//lesion_result_hard//'+path, (lesion_mask[:,:,0]>255*0.3)*255)
        
    print(str(vol),'done')



###
import cv2
import numpy as np
import os
import nibabel as nib
import pandas as pd

lesion_seg_result_root = 'E://submission//AH-net//lesion_result_hard//'
liver_seg_result_root = 'E://submission//AH-net//liver_result_hard//'

cate = dict()
for item in os.listdir(liver_seg_result_root):
    vol = item.split('-')[0]
    if not vol in cate:
        cate[vol] = []
    cate[vol].append(item)

nii_root = 'E://NII//Test'
out_nii_root = 'D://seg_nii4//'
for vol in range(0,70):
    if vol == 59:
        continue
    nii = nib.load(os.path.join(nii_root, 'test-volume-' + str(vol) + '.nii'))
    nii_data = nii.get_data()
    nii_data = np.zeros(nii_data.shape)

    for path in cate[str(vol)]:
        out = np.zeros((512,512))
        sli_num = int(path.split('-')[1][:-4])
        lesion_mask = cv2.imread(os.path.join(lesion_seg_result_root, path))
        liver_mask = cv2.imread(os.path.join(liver_seg_result_root, path))
   
        nii_data[:,:,sli_num-1] = liver_mask[:,:,0]/255+lesion_mask[:,:,0]/255

#    nii_data = nii_data.astype(np.int16)
#    if vol in [8,10,39]:
#        nii_data = (nii_data>0) + 0
#    nii_data = nii_data.astype(np.int16)

    ni_img = nib.Nifti1Image(nii_data, nii.affine, nii.header)
    nib.save(ni_img, out_nii_root+'test-segmentation-{}.nii'.format(str(vol)))
    
    print(vol, 'done!')


