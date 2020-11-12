from utils import seed_everything, RSNADatasetStage1, get_train_transforms, get_valid_transforms, RSNAImgClassifier, RSNAImgClassifierSingle, prepare_train_dataloader, RSNAClassifier, get_stage1_columns

import torch
import catalyst
import time
import pandas as pd 
import numpy as np 
import json
import os
from torch.utils.data.sampler import SequentialSampler, RandomSampler

SEED = 42321

STAGE1_CFGS = [
    {
        'tag': 'efb0_stage1',
        'model_constructor': RSNAImgClassifierSingle,
        'dataset_constructor': RSNADatasetStage1,
        'output_len': 1
    },
    {
        'tag': 'efb0_stage1_multilabel',
        'model_constructor': RSNAImgClassifier,
        'dataset_constructor': RSNADatasetStage1,
        'output_len': 9
    },
]
STAGE1_CFGS_TAG = 'efb0-stage1-single-multi-label'

def rsna_wloss_inference(y_true_img, y_true_exam, y_pred_img, y_pred_exam, chunk_sizes):
    '''
    'negative_exam_for_pe', # exam level 0.0736196319
    'rv_lv_ratio_gte_1', # exam level 0.2346625767
    'rv_lv_ratio_lt_1', # exam level 0.0782208589
    'leftsided_pe', # exam level 0.06257668712
    'chronic_pe', # exam level 0.1042944785
    'rightsided_pe', # exam level 0.06257668712
    'acute_and_chronic_pe', # exam level 0.1042944785
    'central_pe', # exam level 0.1877300613
    'indeterminate' # exam level 0.09202453988
    '''
    
    # transform into torch tensors
    y_true_img, y_true_exam, y_pred_img, y_pred_exam = torch.tensor(y_true_img, dtype=torch.float32), torch.tensor(y_true_exam, dtype=torch.float32), torch.tensor(y_pred_img, dtype=torch.float32), torch.tensor(y_pred_exam, dtype=torch.float32)
    
    # split into chunks (each chunks is for a single exam)
    y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks = torch.split(y_true_img, chunk_sizes, dim=0), torch.split(y_true_exam, chunk_sizes, dim=0), torch.split(y_pred_img, chunk_sizes, dim=0), torch.split(y_pred_exam, chunk_sizes, dim=0)
    
    label_w = torch.tensor([0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988]).view(1, -1)
    img_w = 0.07361963
    bce_func = torch.nn.BCELoss(reduction='none')
    
    total_loss = torch.tensor(0, dtype=torch.float32)
    total_weights = torch.tensor(0, dtype=torch.float32)
    for i, (y_true_img_, y_true_exam_, y_pred_img_, y_pred_exam_) in enumerate(zip(y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks)):
        exam_loss = bce_func(y_pred_exam_[0, :], y_true_exam_[0, :])
        exam_loss = torch.sum(exam_loss*label_w, 1)[0] # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        image_loss = bce_func(y_pred_img_, y_true_img_)
        img_num = chunk_sizes[i]
        qi = torch.sum(y_true_img_)/img_num
        image_loss = torch.sum(img_w*qi*image_loss)
    
        total_loss += exam_loss+image_loss
        total_weights += label_w.sum() + img_w*qi*img_num
        
    final_loss = total_loss/total_weights
    return final_loss

def rsna_wloss(y_true_img, y_true_exam, y_pred_img, y_pred_exam, image_masks, device):
    
    label_w = torch.tensor([0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988]).view(1, -1).to(device)
    img_w = 0.07361963
    bce_func = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    
    total_loss = torch.tensor(0, dtype=torch.float32).to(device)
    total_weights = torch.tensor(0, dtype=torch.float32).to(device)
    for i in range(y_true_img.shape[0]):
        exam_loss = bce_func(y_pred_exam[i, :], y_true_exam[i, :])
        exam_loss = torch.sum(exam_loss*label_w, 1)[0] # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        img_mask = image_masks[i]
        image_loss = bce_func(y_pred_img[i,:], y_true_img[i,:]).flatten()
        image_loss = image_loss*img_mask # mark 0 loss for padding images
        img_num = torch.sum(img_mask) 
        qi = torch.sum(y_true_img[i,:])/img_num
        image_loss = torch.sum(img_w*qi*image_loss)
    
        total_loss += exam_loss+image_loss
        total_weights += label_w.sum() + img_w*qi*img_num
        
    final_loss = total_loss/total_weights
    return final_loss, total_loss, total_weights

def update_stage1_oof_preds(df, cv_df):
    
    res_file_name = STAGE1_CFGS_TAG+"-train.csv"    
    
    new_feats = get_stage1_columns(STAGE1_CFGS)
    for f in new_feats:
        df[f] = 0
    
    if os.path.isfile(res_file_name):
        df = pd.read_csv(res_file_name)
        print('img acc:', ((df[new_feats[0]]>0)==df[CFG['image_target_cols'][0]]).mean())
        return df
    
    
    for fold, (train_fold, valid_fold) in enumerate(zip(CFG['train_folds'], CFG['valid_folds'])):
        if fold < 0:
            continue
            
        valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()
        filt = df.StudyInstanceUID.isin(valid_patients)
        valid_ = df.loc[filt,:].reset_index(drop=True)

        image_preds_all_list = []
        for cfg in STAGE1_CFGS:
            valid_ds = cfg['dataset_constructor'](valid_, 0.0, CFG['train_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=True)

            val_loader = torch.utils.data.DataLoader(
                valid_ds, 
                batch_size=256,
                num_workers=CFG['num_workers'],
                shuffle=False,
                pin_memory=False,
                sampler=SequentialSampler(valid_ds)
            )

            device = torch.device(CFG['device'])
            model = cfg['model_constructor']().to(device)
            model.load_state_dict(torch.load('{}/model_fold_{}_{}'.format(CFG['model_path'], fold, cfg['tag'])))
            model.eval()

            image_preds_all = []
            correct_count = 0
            count = 0
            for step, (imgs, target) in enumerate(val_loader):
                imgs = imgs.to(device).float()
                target = target.to(device).float()

                image_preds = model(imgs)   #output = model(input)

                if len(image_preds.shape) == 1:
                    image_preds = image_preds.view(-1, 1)
                
                correct_count += ((image_preds[:,0]>0) == target[:,0]).sum().detach().item()
                count += imgs.shape[0]
                image_preds_all += [image_preds.cpu().detach().numpy()]
                print('acc: {:.4f}, {}, {}, {}/{}'.format(correct_count/count, correct_count, count, step+1, len(val_loader)), end='\r')
            print()
            
            image_preds_all = np.concatenate(image_preds_all, axis=0)
            image_preds_all_list += [image_preds_all]
        
            del model, val_loader
            torch.cuda.empty_cache()
        
        image_preds_all_list = np.concatenate(image_preds_all_list, axis=1)
        df.loc[filt, new_feats] = image_preds_all_list
        
    df.to_csv(res_file_name, index=False)
    return df

def train_one_epoch(epoch, model, device, scaler, optimizer, train_loader):
    model.train()

    t = time.time()
    loss_sum = 0
    loss_w_sum = 0

    for step, (imgs, per_image_preds, locs, image_labels, exam_label, image_masks) in enumerate(train_loader):
        imgs = imgs.to(device).float()
        per_image_preds = per_image_preds.to(device).float()
        locs = locs.to(device).float()
        image_masks = image_masks.to(device).float()
        image_labels = image_labels.to(device).float()
        exam_label = exam_label.to(device).float()

        with autocast():
            image_preds, exam_pred = model(per_image_preds, locs)   #output = model(input)

            loss, total_loss, total_weights = rsna_wloss(image_labels, exam_label, image_preds, exam_pred, image_masks, device)

            scaler.scale(loss).backward()

            loss_sum += total_loss.detach().item()
            loss_w_sum += total_weights.detach().item()

            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()                

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                print(
                    f'epoch {epoch} train step {step+1}/{len(train_loader)}, ' + \
                    f'loss: {loss_sum/loss_w_sum:.4f}, ' + \
                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(train_loader) else '\n'
                )

def valid_one_epoch(epoch, model, device, scheduler, val_loader, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    loss_w_sum = 0

    for step, (imgs, per_image_preds, locs, image_labels, exam_label, image_masks) in enumerate(val_loader):
        imgs = imgs.to(device).float()
        per_image_preds = per_image_preds.to(device).float()
        locs = locs.to(device).float()
        image_masks = image_masks.to(device).float()
        image_labels = image_labels.to(device).float()
        exam_label = exam_label.to(device).float()

        #print(image_labels.shape, exam_label.shape)
        #with autocast():
        image_preds, exam_pred = model(per_image_preds, locs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        exam_pred, image_preds= post_process(exam_pred, image_preds)

        loss, total_loss, total_weights = rsna_wloss_valid(image_labels, exam_label, image_preds, exam_pred, image_masks, device)

        loss_sum += total_loss.detach().item()
        loss_w_sum += total_weights.detach().item()          

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            print(
                f'epoch {epoch} valid Step {step+1}/{len(val_loader)}, ' + \
                f'loss: {loss_sum/loss_w_sum:.4f}, ' + \
                f'time: {(time.time() - t):.4f}', end='\r' if (step + 1) != len(val_loader) else '\n'
            )
    
    if schd_loss_update:
        scheduler.step(loss_sum/loss_w_sum)
    else:
        scheduler.step()

if __name__ == '__main__':
    with open('config.json') as json_file: 
        CFG = json.load(json_file) 
    
    from  torch.cuda.amp import autocast, GradScaler # for training only, need nightly build pytorch

    seed_everything(SEED)
    
    # read train file
    train_df = pd.read_csv(CFG['train_path'])

    # read cv file
    cv_df = pd.read_csv(CFG['cv_fold_path'])

    with torch.no_grad():
        train_df = update_stage1_oof_preds(train_df,cv_df)
    # img must be sorted before feeding into NN for correct orders

    for fold, (train_fold, valid_fold) in enumerate(zip(CFG['train_folds'], CFG['valid_folds'])):

        train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, train_fold, valid_fold)

        device = torch.device(CFG['device'])
        model = RSNAClassifier(STAGE1_CFGS=STAGE1_CFGS).to(device)
        
        scaler = GradScaler()   
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)

            with torch.no_grad():
                valid_one_epoch(epoch, model, device, scheduler, val_loader, schd_loss_update=schd_loss_update)

        torch.save(model.state_dict(),'{}/model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
        
        model.load_state_dict(torch.load('{}/model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag'])))
        
        # prediction for oof
        valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()
        valid_ = train_df.loc[train_df.StudyInstanceUID.isin(valid_patients),:].reset_index(drop=True)

        with torch.no_grad():
            val_pred_df = inference(model, device, valid_, CFG['train_img_path'])
        
        target = valid_[CFG['image_target_cols']].values
        pred = (val_pred_df[CFG['image_target_cols']].values > 0.5).astype(int)
        print('Image PE Accuracy: {:.3f}'.format((target==pred).mean()*100))
        
        loss = rsna_wloss_inference(valid_[CFG['image_target_cols']].values, valid_[CFG['exam_target_cols']].values, 
                                    val_pred_df[CFG['image_target_cols']].values, val_pred_df[CFG['exam_target_cols']].values, 
                                    list(valid_.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))

        print('Validation loss = {:.4f}'.format(loss.detach().item()))
        
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
        
        train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, np.arange(0, 20), np.array([]))
        device = torch.device(CFG['device'])
        model = RSNAClassifier(STAGE1_CFGS=STAGE1_CFGS).to(device)
        scaler = GradScaler()   
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)

        torch.save(model.state_dict(),'{}/model_{}'.format(CFG['model_path'], CFG['tag']))
