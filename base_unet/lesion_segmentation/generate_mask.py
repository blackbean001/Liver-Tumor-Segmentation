import os 
import cv2
import numpy as np
import torch.optim as optim
from data_util import *
from model import *
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from scipy.misc import imresize
from PIL import Image
from PIL import ImageOps

#root = '/data/liver'
#root = 'E://liver_2dunet//data2//'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model(model, ckpt):
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        print("==> no checkpoint found at '{}'".format(ckpt))

def gen_mask(ckpt, img_path, mask_path):
    
    img = Image.open(img_path) # 3 512 512

    w, h = img.size

    model = uNet(2).cuda()
    model.eval()
    model = load_model(model, ckpt)
    
    img = np.transpose(np.array(img.convert('RGB')).astype(np.float32), [2,0,1])
    img = img[np.newaxis,:,:,:]
    img = torch.from_numpy(img/255)
    data = Variable(img, volatile = True)
    data = data.float().cuda()
    output = model(data)
        
    probs = F.sigmoid(output)
    pred = (probs > 0.5).float()
    pred = pred.data.cpu().numpy()
    pred = pred[0,:,:] * 255

    cv2.imwrite(mask_path, pred)


img_dir = 'E://liver_2dunet//data//volume//test'
lesion_out_mask_dir = 'E://liver_2dunet//test_gaia//dice_liver//results//dice_liver'

if not os.path.isdir(lesion_out_mask_dir):
    os.mkdir(lesion_out_mask_dir)

ckpt = 'E://liver_2dunet//test_gaia//dice_liver//checkpoints//model_best.pth.tar'
exi = os.listdir(lesion_out_mask_dir)
s = 0
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)
    mask_path = os.path.join(lesion_out_mask_dir, img)
    print(img, s)
    if img not in exi:
        gen_mask(ckpt, img_path, mask_path)
    s += 1

