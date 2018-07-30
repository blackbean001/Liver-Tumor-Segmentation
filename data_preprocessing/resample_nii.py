# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:22:39 2018

@author: jasonsli
"""

import nibabel as nib
import os
import numpy as np
import cv2
import scipy.ndimage

#NII_root = 'E://NII//NII//Training'
#for path in os.listdir(NII_root):
#    if path.startswith('v') and path.endswith('nii'):
#        img = nib.load(os.path.join(NII_root, path))
#        hdr = img.header
#        print(path, hdr['pixdim'][1:4])

# calculate averate z-axis slice-space relationship
NII_root = 'E://NII//NII//Training'

none_liver_mask_root = 'D://process_data//pngdata//mask//liver_mask_none'
cate = dict()
for item in os.listdir(none_liver_mask_root):
    vol = item.split('-')[0]
    if not vol in cate:
        cate[vol] = []
    cate[vol].append(int(item.split('-')[1][:-4]))

length = []
abnormal = []
for i in range(131):
    num_slice = len(cate[str(i)])

    seg_name = 'segmentation-{}.nii'.format(str(i))
    vol_name = 'volume-{}.nii'.format(str(i))

    label = nib.load(os.path.join(NII_root, seg_name))
    img = nib.load(os.path.join(NII_root, vol_name))

    hdr = img.header
    spacing = hdr['pixdim'][1:4]
    if (spacing[0], spacing[1], spacing[2]) == (1,1,1):
        abnormal.append(i)
    if (spacing[0], spacing[1], spacing[2]) != (1,1,1):
        length.append(num_slice * spacing[2])    
ave_liver_length = np.mean(length)

for i in abnormal:
    num_slice = len(cate[str(i)])

    seg_name = 'segmentation-{}.nii'.format(str(i))
    vol_name = 'volume-{}.nii'.format(str(i))
    
    print(ave_liver_length/(num_slice+.0))

# save real z pixdim for all volumes    
real_z_pixdim = dict()
for i in range(131):
    num_slice = len(cate[str(i)])
    seg_name = 'segmentation-{}.nii'.format(str(i))
    vol_name = 'volume-{}.nii'.format(str(i))

    if i in abnormal:
        real_z_pixdim[str(i)] = ave_liver_length/(num_slice+.0)
    if i not in abnormal:
        img = nib.load(os.path.join(NII_root, vol_name))
        hdr = img.header
        real_z_pixdim[str(i)] = float(hdr['pixdim'][3])

# change to new z, resample z 
for path in os.listdir(NII_root):
    if path.startswith('s'):
        continue
 
    vol = path.split('-')[1][:-4]
    spacing_z = real_z_pixdim[vol]
    
    img = nib.load(os.path.join(NII_root, path))
    hdr = img.header
    
    spacing = hdr['pixdim'][1:4]
    spacing[2] = spacing_z
    new_spacing = np.array([spacing[0], spacing[1], 1])

    resize_factor = spacing / new_spacing
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / img.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(img.get_data().astype(np.float), real_resize_factor, mode='nearest')

    depth = image.shape[2]
  
    image[image < -150] = -150
    image[image > 250] = 250
    
    for i in range(depth):
        img = image[:,:,i]
        img = (img - np.min(img))/(np.max(img)-np.min(img))*255.0
        cv2.imwrite('D://process_data//resampled//' + 'vol//' + vol + '-' + str(i)+'.png',img)

    print(path)
    
for path in os.listdir(NII_root):
    if path.startswith('v'):
        continue
 
    vol = path.split('-')[1][:-4]
    spacing_z = real_z_pixdim[vol]
    
    img = nib.load(os.path.join(NII_root, path))
    hdr = img.header
    
    spacing = hdr['pixdim'][1:4]
    spacing[2] = spacing_z
    new_spacing = np.array([spacing[0], spacing[1], 1])

    resize_factor = spacing / new_spacing
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / img.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(img.get_data().astype(np.uint16), real_resize_factor, mode='nearest')

    depth = image.shape[2]
    
    for i in range(depth):
        # generate liver mask
        img = 255*((image[:,:,i]>0)+.0)
        cv2.imwrite('D://process_data//resampled//' + 'liver_mask//' + vol + '-' + str(i)+'.png',img)
        # generate lesion mask
        img2 = 255*((image[:,:,i]==2)+.0)
        cv2.imwrite('D://process_data//resampled//' + 'lesion_mask//' + vol + '-' + str(i)+'.png',img2)
        
    print(path)

def register(root):
    for path in os.listdir(root):
        vol = path.split('-')[0]
    
        if int(vol) <= 52:
            img_vol = cv2.imread(os.path.join(root, path))
            img_vol = cv2.flip(img_vol, 1)
            cv2.imwrite(os.path.join(root, path), img_vol)

        if int(vol) >= 68 and int(vol) <= 82:
            img_vol = cv2.imread(os.path.join(root, path))
            img_vol = cv2.flip(img_vol, -1)
            cv2.imwrite(os.path.join(root, path), img_vol)    

roots = ['D://process_data//resampled//' + 'liver_mask//', 'D://process_data//resampled//' + 'lesion_mask//', 'D://process_data//resampled//' + 'vol//']
for root in roots:
    register(root)
    print(root)



