import torch
from torch import nn
import os 
import numpy as np
import random
import albumentations as albu
import pandas as pd 
import pydicom
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset,DataLoader
import json

with open('config.json') as json_file: 
    CFG = json.load(json_file) 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    return X

def get_img(path, transforms, opencv=False):
    
    d = pydicom.read_file(path)
    '''
    res = cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (CFG['img_size'], CFG['img_size'])), d.ImagePositionPatient[2]
    '''

    '''
    RED channel / LUNG window / level=-600, width=1500
    GREEN channel / PE window / level=100, width=700
    BLUE channel / MEDIASTINAL window / level=40, width=400
    '''
    
    img = (d.pixel_array * d.RescaleSlope) + d.RescaleIntercept
    
    r = window(img, -600, 1500)
    g = window(img, 100, 700)
    b = window(img, 40, 400)
    
    res = np.concatenate([r[:, :, np.newaxis],
                          g[:, :, np.newaxis],
                          b[:, :, np.newaxis]], axis=-1)
    
    if opencv:
        res = transforms(image=(res*255.0).astype('uint8'))['image']
    #res = cv2.resize(res, (CFG['img_size'], CFG['img_size']))
        res = torch.div(res, 255.)
    else:
        res = transforms(image=res)['image']
    return res

def get_train_transforms():
    return albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.pytorch.ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return albu.Compose([
            albu.Resize(256, 256),
            albu.pytorch.ToTensorV2(p=1.0),
        ], p=1.0)

def get_stage1_columns():
    new_feats = []
    for cfg in STAGE1_CFGS: # CHECK THIS OUT, DOES IT WORK
        for i in range(cfg['output_len']):
            f = cfg['tag']+'_'+str(i)
            new_feats += [f]
        
    return new_feats

class RSNADatasetStage1(Dataset):
    '''
    Stage 1 dataset for multilabel (exam level)
    '''
    def __init__(
        self, df, label_smoothing, data_root, 
        image_subsampling=True, transforms=None, output_label=True, opencv=False
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.label_smoothing = label_smoothing
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        self.opencv = opencv
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.df[CFG['image_target_cols']].values[index]
            target[1:-1] = target[0]*target[1:-1] # if PE == 1, keep the original label; otherwise clean to 0 (except indeterminate)
            
        path = "{}/{}/{}/{}.dcm".format(self.data_root, 
                                        self.df.iloc[index]['StudyInstanceUID'], 
                                        self.df.iloc[index]['SeriesInstanceUID'], 
                                        self.df.iloc[index]['SOPInstanceUID'])

        img = get_img(path, self.transforms, self.opencv)

        if self.output_label == True:
            target = np.clip(target, self.label_smoothing, 1 - self.label_smoothing)    
            return img, target
        else:
            return img

class RSNADataset(Dataset):
    '''
    Dataset for extracting from embeddings
    '''
    def __init__(
        self, df, label_smoothing, data_root, 
        image_subsampling=True, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df
        self.patients = self.df['StudyInstanceUID'].unique()
        self.image_subsampling = image_subsampling
        self.label_smoothing = label_smoothing
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        
    def get_patients(self):
        return self.patients
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index: int):
        
        patient = self.patients[index]
        df_ = self.df.loc[self.df.StudyInstanceUID == patient]
        
        per_image_feats = get_stage1_columns()
        
        if self.image_subsampling:
            img_num = min(CFG['img_num'], df_.shape[0])
            
            # naive image subsampling
            img_ix = np.random.choice(np.arange(df_.shape[0]), replace=False, size=img_num)
            
            # get all images, then slice location and sort according to z values
            imgs = np.zeros((CFG['img_num'],), np.float32)
            per_image_preds = np.zeros((CFG['img_num'], len(per_image_feats)), np.float32)
            locs = np.zeros((CFG['img_num'],), np.float32)
            image_masks = np.zeros((CFG['img_num'],), np.float32)
            image_masks[:img_num] = 1.
            
            # get labels
            if self.output_label:
                exam_label = df_[CFG['exam_target_cols']].values[0]
                image_labels = np.zeros((CFG['img_num'], len(CFG['image_target_cols'])), np.float32)
            
        else:
            img_num = df_.shape[0]
            img_ix = np.arange(df_.shape[0])
            
            # get all images, then slice location and sort according to z values
            imgs = np.zeros((img_num, ), np.float32)
            per_image_preds = np.zeros((img_num, len(per_image_feats)), np.float32)
            locs = np.zeros((img_num,), np.float32)
            image_masks = np.zeros((img_num,), np.float32)
            image_masks[:img_num] = 1.
            
            # get labels
            if self.output_label:
                exam_label = df_[CFG['exam_target_cols']].values[0]
                image_labels = np.zeros((img_num, len(CFG['image_target_cols'])), np.float32)
                
        for i, im_ix in enumerate(img_ix):
            path = "{}/{}/{}/{}.dcm".format(self.data_root, 
                                            df_['StudyInstanceUID'].values[im_ix], 
                                            df_['SeriesInstanceUID'].values[im_ix], 
                                            df_['SOPInstanceUID'].values[im_ix])
            
            d = pydicom.read_file(path)
            locs[i] = d.ImagePositionPatient[2]
            per_image_preds[i,:] = df_[per_image_feats].values[im_ix,:]
            
            if self.output_label == True:
                image_labels[i] = df_[CFG['image_target_cols']].values[im_ix]

        seq_ix = np.argsort(locs)
        
        locs = locs[seq_ix]
        locs[1:img_num] = locs[1:img_num]-locs[0:img_num-1]
        locs[0] = 0
        
        per_image_preds = per_image_preds[seq_ix]
        
        # patient level features: 1
        
        # train, train-time valid, multiple patients: imgs, locs, image_labels, exam_label, img_num
        # whole valid-time valid, single patient: imgs, locs, image_labels, exam_label, img_num, sorted id
        # whole test-time test, single patient: imgs, locs, img_num, sorted_id
        
        # do label smoothing
        if self.output_label == True:
            image_labels = image_labels[seq_ix]
            image_labels = np.clip(image_labels, self.label_smoothing, 1 - self.label_smoothing)
            exam_label =  np.clip(exam_label, self.label_smoothing, 1 - self.label_smoothing)
            
            return imgs, per_image_preds, locs, image_labels, exam_label, image_masks
        else:
            return imgs, per_image_preds, locs, img_num, index, seq_ix

class RNSAImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = EfficientNet.from_name(CFG['efbnet'])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
    def get_dim(self):
        return self.cnn_model._fc.in_features
        
    def forward(self, x):
        feats = self.cnn_model.extract_features(x)
        return self.pooling(feats).view(x.shape[0], -1)   

class RSNAImgClassifierSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = RNSAImageFeatureExtractor()
        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 1)
        
    def forward(self, imgs):
        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size
        image_preds = self.image_predictors(imgs_embdes)
        
        return image_preds

class RSNAImgClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = RNSAImageFeatureExtractor()
        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 9)
        
    def forward(self, imgs):
        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size
        image_preds = self.image_predictors(imgs_embdes)
        
        return image_preds