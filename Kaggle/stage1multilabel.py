package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
utils_path = '../input/rsna-str-github/'
import sys; sys.path.append(package_path); sys.path.append(utils_path)

bash_commands = [
            'cp ../input/gdcm-conda-install/gdcm.tar .',
            'tar -xvzf gdcm.tar',
            'conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2',
            'cp ../input/rsna-str-github/config.json .',
            ]

import subprocess
for bashCommand in bash_commands:
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

from utils import seed_everything, RSNADatasetStage1, get_train_transforms, get_valid_transforms, RSNAImgClassifier, valid_one_epoch, prepare_train_dataloader

import torch
import catalyst
import time
import pandas as pd 
import numpy as np 
import json

SEED = 42321

def rsna_wloss(y_true_img, y_pred_img, device):
    bce_func = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    y_pred_img = y_pred_img.view(*y_true_img.shape)
    image_loss = bce_func(y_pred_img, y_true_img)
    correct_count = ((y_pred_img>0) == y_true_img).sum(axis=0)
    counts = y_true_img.size()[0]
    return image_loss, correct_count, counts

def train_one_epoch(epoch, model, device, scaler, optimizer, train_loader):
    model.train()

    t = time.time()
    loss_sum = 0
    acc_sum = None
    loss_w_sum = 0
    acc_record = []
    loss_record = []
    avg_cnt = 40
    
    for step, (imgs, image_labels) in enumerate(train_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()

        with autocast():
            image_preds = model(imgs)   #output = model(input)

            image_loss, correct_count, counts = rsna_wloss(image_labels, image_preds, device)
            
            loss = image_loss/counts
            scaler.scale(loss).backward()

            loss_ = image_loss.detach().item()/counts
            acc_ = correct_count.detach().cpu().numpy()/counts
            
            loss_record += [loss_]
            acc_record += [acc_]
            loss_record = loss_record[-avg_cnt:]
            acc_record = acc_record[-avg_cnt:]
            loss_sum = np.vstack(loss_record).mean(axis=0)
            acc_sum = np.vstack(acc_record).mean(axis=0)
            
            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()                

            acc_details = ["{:.5}: {:.4f}".format(f, float(acc_sum[i])) for i, f in enumerate(CFG['image_target_cols'])]
            acc_details = ", ".join(acc_details)

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                print(
                    f'epoch {epoch} train Step {step+1}/{len(train_loader)}, ' + \
                    f'loss: {loss_sum[0]:.3f}, ' + \
                    acc_details + ', ' + \
                    f'time: {(time.time() - t):.2f}', end='\r' if (step + 1) != len(train_loader) else '\n'
                )

if __name__ == '__main__':
    with open('config.json') as json_file: 
        CFG = json.load(json_file) 
    if CFG['train']:
        from torch.cuda.amp import autocast, GradScaler # for training only, need nightly build pytorch

    seed_everything(SEED)
    
    if CFG['train']:
        # read train file
        train_df = pd.read_csv(CFG['train_path'])

        # read cv file
        cv_df = pd.read_csv(CFG['cv_fold_path'])

        # img must be sorted before feeding into NN for correct orders
    else:
        test_df = pd.read_csv(CFG['test_path'])
    
    if CFG['train']:
        for fold, (train_fold, valid_fold) in enumerate(zip(CFG['train_folds'], CFG['valid_folds'])):
            if fold < 0:
                continue
            print('Fold:', fold)   
            train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, train_fold, valid_fold)

            device = torch.device(CFG['device'])
            model = RSNAImgClassifier().to(device)
            scaler = GradScaler()   
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False
            
            for epoch in range(CFG['epochs']):
                train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)
                
                with torch.no_grad():
                    valid_one_epoch(epoch, model, device, scheduler, val_loader, schd_loss_update=schd_loss_update)
            
            torch.save(model.state_dict(),'{}/model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
            del model, optimizer, train_loader, val_loader, scaler, scheduler
            torch.cuda.empty_cache()

        # Train on all data after val
        '''
        train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, np.arange(0, 20), np.array([]))
        device = torch.device(CFG['device'])
        model = RSNAImgClassifier().to(device)
        scaler = GradScaler()   
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)

        torch.save(model.state_dict(),'{}/model_{}'.format(CFG['model_path'], CFG['tag']))
        '''
    else:
        assert False