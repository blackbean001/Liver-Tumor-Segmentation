import nibabel as nib
import numpy as np
import os
import cv2

volume_root = 'F://process_data//niidata//volume//'
# volume_test_root = 'D://liver//MHD//Test//volume//'
mask_root = 'F://process_data//niidata//mask//'
# mask_test_root = 'D://liver//MHD//Test//volume//'

volume_out_train_root = 'F://process_data//pngdata//train//volume//'
mask_out_train_root = 'F://process_data//pngdata//train//mask//'
mask_out_train_root_liver = 'F://process_data//pngdata//train//mask//liver_mask//'
mask_out_train_root_lesion = 'F://process_data//pngdata//train//mask//lesion_mask//'

volume_out_test_root = 'F://process_data//pngdata//test//volume//'
mask_out_test_root = 'F://process_data//pngdata//test//mask//'
mask_out_test_root_liver = 'F://process_data//pngdata//test//mask//liver_mask//'
mask_out_test_root_lesion = 'F://process_data//pngdata//test//mask//lesion_mask//'

#for i in [volume_out_root, mask_out_root_liver, mask_out_root_lesion]:
#    if not os.path.exists(i):
#        os.mkdir(i)

def to_png(nii_path):
    img_array = nib.load(nii_path).get_data()
    img_array[img_array < -150] = -150
    img_array[img_array > 250] = 250
        
    a, b, depth = img_array.shape
    
    output = []
    for i in range(depth):
        out = cv2.merge([img_array[:,:,i],img_array[:,:,i],img_array[:,:,i]])

        cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
        
        output.append(out)
    return output

def mask_to_png(nii_path):
    img_array = nib.load(nii_path).get_data()
    
    a, b, depth = img_array.shape

    output1 = []
    output2 = []
    for i in range(depth):
        out = img_array[:, :, i]
        
        out1 = np.zeros((a, b))
        out2 = np.zeros((a, b))
        
        l1 = np.where(out == 1)
        l2 = np.where(out == 2)
        
        for j in range(len(l1[0])):
            out1[l1[0][j], l1[1][j]] = 255
        for j in range(len(l2[0])):
            out1[l2[0][j], l2[1][j]] = 255
            out2[l2[0][j], l2[1][j]] = 255
                
        output1.append(out1)
        output2.append(out2)
        
    return output1, output2


for path in os.listdir(volume_root):
    vol = path.split('-')[1].split('.')[0]
    output = to_png(os.path.join(volume_root, path))
    if int(vol) < 105:
        index = 1
        for out in output:
            cv2.imwrite(volume_out_train_root + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
    if int(vol) >= 105:
        index = 1
        for out in output:
            cv2.imwrite(volume_out_test_root + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1        

for path in os.listdir(mask_root):
    vol = path.split('-')[1].split('.')[0]
    output1, output2 = mask_to_png(os.path.join(mask_root, path))
    if int(vol) < 105:
        index = 1
        for out in output1:
            cv2.imwrite(mask_out_train_root_liver + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
        index = 1
        for out in output2:
            cv2.imwrite(mask_out_train_root_lesion + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
    if int(vol) >= 105:
        index = 1
        for out in output1:
            cv2.imwrite(mask_out_test_root_liver + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
        index = 1
        for out in output2:
            cv2.imwrite(mask_out_test_root_lesion + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
    print(vol,'done!')

            
'''
change the directions of train images to that of test images
0-52: cv2.flip(1)
68-82: cv2.flip(-1)
'''

for path in os.listdir(volume_out_train_root):
    vol = path.split('-')[0]
    
    if int(vol) <= 52:
        img_vol = cv2.imread(os.path.join(volume_out_train_root, path))
        img_vol = cv2.flip(img_vol, 1)
        cv2.imwrite(os.path.join(volume_out_train_root, path), img_vol)
        
        mask_liver = cv2.imread(os.path.join(mask_out_train_root_liver, path))
        mask_liver = cv2.flip(mask_liver, 1)
        cv2.imwrite(os.path.join(mask_out_train_root_liver, path), mask_liver)
        
        mask_lesion = cv2.imread(os.path.join(mask_out_train_root_lesion, path))
        mask_lesion = cv2.flip(mask_lesion, 1)
        cv2.imwrite(os.path.join(mask_out_train_root_lesion, path), mask_lesion)

    if int(vol) >= 68 and int(vol) <= 82:
        img_vol = cv2.imread(os.path.join(volume_out_train_root, path))
        img_vol = cv2.flip(img_vol, -1)
        cv2.imwrite(os.path.join(volume_out_train_root, path), img_vol)
        
        mask_liver = cv2.imread(os.path.join(mask_out_train_root_liver, path))
        mask_liver = cv2.flip(mask_liver, -1)
        cv2.imwrite(os.path.join(mask_out_train_root_liver, path), mask_liver)
        
        mask_lesion = cv2.imread(os.path.join(mask_out_train_root_lesion, path))
        mask_lesion = cv2.flip(mask_lesion, -1)
        cv2.imwrite(os.path.join(mask_out_train_root_lesion, path), mask_lesion)
        
    print(vol,'done!')


img_root = 'F://process_data//pngdata//train//volume'
img_out_root = 'F://process_data//pngdata//train//volume_none'

mask_root = 'F://process_data//pngdata//train//mask//liver_mask'
mask_out_root = 'F://process_data//pngdata//train//mask//liver_mask_none'

for path in os.listdir(mask_root):
    mask = cv2.imread(os.path.join(mask_root, path))
    img = cv2.imread(os.path.join(img_root, path))
    if np.sum(mask) > 0:
        cv2.imwrite(os.path.join(mask_out_root, path), mask)
        cv2.imwrite(os.path.join(img_out_root, path), img)

         

#######################################################################################################################

'''
to 3-channels images

'''
import nibabel as nib
import numpy as np
import os
import cv2

volume_root = 'D://liver//NII//Training//volume//'
mask_root = 'D://liver//NII//Training//mask//'

volume_out_train_root = 'F://process_data//pngdata//colored//train//volume//'
mask_out_train_root = 'F://process_data//pngdata//colored//train//mask//'
mask_out_train_root_liver = 'F://process_data//pngdata//colored//train//mask//liver_mask//'
mask_out_train_root_lesion = 'F://process_data//pngdata//colored//train//mask//lesion_mask//'

volume_out_test_root = 'F://process_data//pngdata//colored//test//volume//'
mask_out_test_root = 'F://process_data//pngdata//colored//test//mask//'
mask_out_test_root_liver = 'F://process_data//pngdata//colored//test//mask//liver_mask//'
mask_out_test_root_lesion = 'F://process_data//pngdata//colored//test//mask//lesion_mask//'

#for i in [volume_out_root, mask_out_root_liver, mask_out_root_lesion]:
#    if not os.path.exists(i):
#        os.mkdir(i)

def to_png(nii_path):
    img_array = nib.load(nii_path).get_data()
    img_array[img_array < -150] = -150
    img_array[img_array > 250] = 250
    
#    ind = nib.load(nii_path).header['srow_x'][0]
#    img_array = 255*(img_array - np.min(img_array))/(np.max(img_array) - np.min(img_array))

    a, b, depth = img_array.shape
    
    output = []
    for i in range(depth):
        out = np.zeros((a, b, 3))
        if i == 0:
            out[:, :, 0] = img_array[:, :, 0]
            out[:, :, 1] = img_array[:, : ,1]
            out[:, :, 2] = img_array[:, :, 2]
        if i == (depth - 1):
            out[:, :, 0] = img_array[:, :, i-2]
            out[:, :, 1] = img_array[:, :, i-1]
            out[:, :, 2] = img_array[:, :, i]
        if i != 0 and i != (depth-1):
            out[:, :, 0] = img_array[:, :, i-1]
            out[:, :, 1] = img_array[:, :, i]
            out[:, :, 2] = img_array[:, :, i+1]
               
        cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
        
        output.append(out)
    return output

def mask_to_png(nii_path):
    img_array = nib.load(nii_path).get_data()
    
    a, b, depth = img_array.shape

    output1 = []
    output2 = []
    for i in range(depth):
        out = img_array[:, :, i]
        
        out1 = np.zeros((a, b))
        out2 = np.zeros((a, b))
        
        l1 = np.where(out == 1)
        l2 = np.where(out == 2)

        for j in range(len(l1[0])):
            out1[l1[0][j], l1[1][j]] = 255
        for j in range(len(l2[0])):
            out1[l2[0][j], l2[1][j]] = 255
            out2[l2[0][j], l2[1][j]] = 255

        output1.append(out1)
        output2.append(out2)
        
    return output1, output2


for path in os.listdir(volume_root):
    vol = path.split('-')[1].split('.')[0]
    output = to_png(os.path.join(volume_root, path))
    if int(vol) < 105:
        index = 1
        for out in output:
            cv2.imwrite(volume_out_train_root + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
    if int(vol) >= 105:
        index = 1
        for out in output:
            cv2.imwrite(volume_out_test_root + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1


for path in os.listdir(mask_root):
    vol = path.split('-')[1].split('.')[0]
    output1, output2 = mask_to_png(os.path.join(mask_root, path))
    if int(vol) < 105:
        index = 1
        for out in output1:
            cv2.imwrite(mask_out_train_root_liver + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
        index = 1
        for out in output2:
            cv2.imwrite(mask_out_train_root_lesion + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
    if int(vol) >= 105:
        index = 1
        for out in output1:
            cv2.imwrite(mask_out_test_root_liver + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1
        index = 1
        for out in output2:
            cv2.imwrite(mask_out_test_root_lesion + path.split('.')[0].split('-')[1] + '-' + str(index) + '.png' , out)
            index += 1            
            
            
'''
change the directions of train images to that of test images
0-52: cv2.flip(1)
68-82: cv2.flip(-1)
'''

for path in os.listdir(volume_out_train_root):
    vol = path.split('-')[0]
    
    if int(vol) <= 52:
        img_vol = cv2.imread(os.path.join(volume_out_train_root, path))
        img_vol = cv2.flip(img_vol, 1)
        cv2.imwrite(os.path.join(volume_out_train_root, path), img_vol)
        
        mask_liver = cv2.imread(os.path.join(mask_out_train_root_liver, path))
        mask_liver = cv2.flip(mask_liver, 1)
        cv2.imwrite(os.path.join(mask_out_train_root_liver, path), mask_liver)
        
        mask_lesion = cv2.imread(os.path.join(mask_out_train_root_lesion, path))
        mask_lesion = cv2.flip(mask_lesion, 1)
        cv2.imwrite(os.path.join(mask_out_train_root_lesion, path), mask_lesion)

    if int(vol) >= 68 and int(vol) <= 82:
        img_vol = cv2.imread(os.path.join(volume_out_train_root, path))
        img_vol = cv2.flip(img_vol, -1)
        cv2.imwrite(os.path.join(volume_out_train_root, path), img_vol)
        
        mask_liver = cv2.imread(os.path.join(mask_out_train_root_liver, path))
        mask_liver = cv2.flip(mask_liver, -1)
        cv2.imwrite(os.path.join(mask_out_train_root_liver, path), mask_liver)
        
        mask_lesion = cv2.imread(os.path.join(mask_out_train_root_lesion, path))
        mask_lesion = cv2.flip(mask_lesion, -1)
        cv2.imwrite(os.path.join(mask_out_train_root_lesion, path), mask_lesion)
        
        
img_root = 'F://process_data//pngdata//colored//train//volume'
img_out_root = 'F://process_data//pngdata//colored//train//volume_none'

mask_root = 'F://process_data//pngdata//colored//train//mask//liver_mask'
mask_out_root = 'F://process_data//pngdata//colored//train//mask//liver_mask_none'

for path in os.listdir(mask_root):
    mask = cv2.imread(os.path.join(mask_root, path))
    img = cv2.imread(os.path.join(img_root, path))
    if np.sum(mask) > 0:
        cv2.imwrite(os.path.join(mask_out_root, path), mask)
        cv2.imwrite(os.path.join(img_out_root, path), img)