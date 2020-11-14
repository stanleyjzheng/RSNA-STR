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
from albumentations.pytorch import ToTensorV2
import json
import time

with open('config.json') as json_file: 
    CFG = json.load(json_file) 

def get_valid_transforms():
    return albu.Compose([
        albu.Resize(CFG["img_size"], CFG["img_size"], p=1.0),
        ToTensorV2(p=1.0),
        ], p=1.0)

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
    '''
    Retrive RGB image from dicom files
    RED channel / LUNG window / level=-600, width=1500
    GREEN channel / PE window / level=100, width=700
    BLUE channel / MEDIASTINAL window / level=40, width=400
    Input:
    path - str; image path
    transforms - albu.Compose; containing desired transformations
    '''
    d = pydicom.read_file(path)
    img = (d.pixel_array * d.RescaleSlope) + d.RescaleIntercept
    
    r = window(img, -600, 1500)
    g = window(img, 100, 700)
    b = window(img, 40, 400)
    
    res = np.concatenate([r[:, :, np.newaxis],
                          g[:, :, np.newaxis],
                          b[:, :, np.newaxis]], axis=-1)
    
    if opencv:
        res = transforms(image=(res*255.0).astype('uint8'))['image']
        res = torch.div(res, 255.)
    else:
        res = transforms(image=res)['image']
    return res

def get_stage1_columns(STAGE1_CFGS):
    new_feats = []
    for cfg in STAGE1_CFGS:
        for i in range(cfg['output_len']):
            f = cfg['tag']+'_'+str(i)
            new_feats += [f]
        
    return new_feats

def valid_one_epoch(epoch, model, device, scheduler, val_loader, loss_fn=None, schd_loss_update=False):
    '''
    Validation for stage 1 models (untested)
    '''
    model.eval()

    t = time.time()
    loss_sum = 0
    acc_sum = 0
    loss_w_sum = 0

    for step, (imgs, image_labels) in enumerate(val_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()
        
        image_preds = model(imgs)

        image_loss, correct_count, counts = loss_fn(image_labels, image_preds, device)

        loss = image_loss/counts
        
        loss_sum += image_loss.detach().item()
        acc_sum += correct_count.detach().cpu().numpy()
        loss_w_sum += counts     
        if isinstance(acc_sum, np.ndarray):
            acc_details = ["{:.5}: {:.4f}".format(f, acc_sum[i]/loss_w_sum) for i, f in enumerate(CFG['image_target_cols'])]
            acc_details = ", ".join(acc_details)
        else: 
            acc_details = "Accuracy: {:.4f}".format(acc_sum/loss_w_sum)
        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            print(
                f'epoch {epoch} valid Step {step+1}/{len(val_loader)}, ' + \
                f'loss: {loss_sum/loss_w_sum:.3f}, ' + \
                acc_details + ', ' + \
                f'time: {(time.time() - t):.2f}', end='\r' if (step + 1) != len(val_loader) else '\n'
            )
    
    if schd_loss_update:
        scheduler.step(loss_sum/loss_w_sum)
    else:
        scheduler.step()

class RSNADatasetStage1(Dataset):
    '''
    Stage 1 dataset
    Input:
    opencv - If openCV based augmentations are used (pixel values must be betseen 0, 255)
    image_label - If False, return 9 exam-level labels, otherwise return single image level label
    '''
    def __init__(
        self, df, label_smoothing, data_root, 
        image_subsampling=True, transforms=None, output_label=True, opencv=False, image_label=False
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.label_smoothing = label_smoothing
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        self.opencv = opencv
        self.image_label = image_label
        self.image_subsampling = image_subsampling
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            if self.image_label:
                target = self.df.iloc[index][CFG['image_target_cols'][0]]
            else:
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
    Dataset containing extracted embeddings and corresponding labels.
    Used in RNN training and inference
    '''
    def __init__(
        self, df, label_smoothing, data_root, 
        image_subsampling=True, transforms=None, output_label=True, STAGE1_CFGS=None
    ):
        
        super().__init__()
        self.df = df
        self.patients = self.df['StudyInstanceUID'].unique()
        self.image_subsampling = image_subsampling
        self.label_smoothing = label_smoothing
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        self.STAGE1_CFGS = STAGE1_CFGS
        
    def get_patients(self):
        return self.patients
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index: int):
        
        patient = self.patients[index]
        df_ = self.df.loc[self.df.StudyInstanceUID == patient]
        
        per_image_feats = get_stage1_columns(self.STAGE1_CFGS)
        
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

class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        x_size= x.size()
        c_in = x.contiguous().view(x_size[0] * x_size[1], *x_size[2:])
        
        c_out = self.module(c_in)
        r_in = c_out.view(x_size[0], x_size[1], -1)
        if self.batch_first is False:
            r_in = r_in.permute(1, 0, 2)
        return r_in 

class RSNAClassifier(nn.Module):
    '''
    Inference/stage2 version of model
    '''
    def __init__(self, hidden_size=64, STAGE1_CFGS=None):
        super().__init__()
        
        self.gru = nn.GRU(len(get_stage1_columns(STAGE1_CFGS))+1, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        
        self.image_predictors = TimeDistributed(nn.Linear(hidden_size*2, 1))
        self.exam_predictor = nn.Linear(hidden_size*2*2, 9)
        
    def forward(self, img_preds, locs):
        
        embeds = torch.cat([img_preds, locs.view(locs.shape[0], locs.shape[1], 1)], dim=2) # bs * ts * fs
        
        embeds, _ = self.gru(embeds)
        image_preds = self.image_predictors(embeds)
        
        avg_pool = torch.mean(embeds, 1)
        max_pool, _ = torch.max(embeds, 1)
        conc = torch.cat([avg_pool, max_pool], 1)
        
        exam_pred = self.exam_predictor(conc)
        return image_preds, exam_pred

class RNSAImageFeatureExtractor(nn.Module):
    '''
    Loads convolutional layers of efficientnet to use as feature extractor for RNN
    Used in inference/stage2
    '''
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
    '''
    Initializes Stage 1 image level model
    '''
    def __init__(self):
        super().__init__()
        self.cnn_model = RNSAImageFeatureExtractor()
        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 1)
        
    def forward(self, imgs):
        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size
        image_preds = self.image_predictors(imgs_embdes)
        
        return image_preds

class RSNAImgClassifier(nn.Module):
    '''
    Initializes Stage 1 multilabel model
    '''
    def __init__(self):
        super().__init__()
        self.cnn_model = RNSAImageFeatureExtractor()
        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 9)
        
    def forward(self, imgs):
        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size
        image_preds = self.image_predictors(imgs_embdes)
        
        return image_preds

def prepare_train_dataloader(train, cv_df, train_fold, valid_fold, train_transforms, image_label=False):
    '''
    Prepares Stage 1 dataset. Note that set pin_memory=True will be faster if memory is adequate
    Inputs:
    train - dataframe; train data from train.csv
    cv_df - dataframe; cross validation with folds
    train_fold - list of int; list of trianing folds corresponding to cv_df
    valid_fold - list of int; list of validiation folds corresponding to cv_df
    '''
    from catalyst.data.sampler import BalanceClassSampler
    
    train_patients = cv_df.loc[cv_df.fold.isin(train_fold), 'StudyInstanceUID'].unique()
    valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()

    train_ = train.loc[train.StudyInstanceUID.isin(train_patients),:].reset_index(drop=True)
    valid_ = train.loc[train.StudyInstanceUID.isin(valid_patients),:].reset_index(drop=True)

    # train mode to do image-level subsampling
    train_ds = RSNADatasetStage1(train_, 0.0, CFG['train_img_path'],  image_subsampling=False, transforms=train_transforms(), output_label=True, image_label=image_label, opencv=True) 
    valid_ds = RSNADatasetStage1(valid_, 0.0, CFG['train_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=True, image_label=image_label)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=CFG['num_workers'],
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False, 
    )
    return train_loader, val_loader

def prepare_stage2_train_dataloader(train, cv_df, train_fold, valid_fold, train_transforms, STAGE1_CFGS):
    
    train_patients = cv_df.loc[cv_df.fold.isin(train_fold), 'StudyInstanceUID'].unique()
    valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()

    train_ = train.loc[train.StudyInstanceUID.isin(train_patients),:].reset_index(drop=True)
    valid_ = train.loc[train.StudyInstanceUID.isin(valid_patients),:].reset_index(drop=True)

    # train mode to do image-level subsampling
    train_ds = RSNADataset(train_, 0.0, CFG['train_img_path'], STAGE1_CFGS=STAGE1_CFGS, image_subsampling=True, transforms=train_transforms(), output_label=True) 
    valid_ds = RSNADataset(valid_, 0.0, CFG['train_img_path'], STAGE1_CFGS=STAGE1_CFGS, image_subsampling=False, transforms=get_valid_transforms(), output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=True,        
        num_workers=CFG['num_workers'],
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['train_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader