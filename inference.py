from utils import seed_everything, RSNADatasetStage1, get_train_transforms, get_valid_transforms, RSNAImgClassifier, valid_one_epoch, prepare_train_dataloader, RNSAImageFeatureExtractor

import torch
import catalyst
import time
import pandas as pd 
import numpy as np 
import json
import albumentations as albu

do_full = False
SEED = 42321

def get_test_augmentation(resize_to=(256,256)):
    test_transform = [
        albu.Resize(*resize_to),
    ]
    return albu.Compose(test_transform) 

def post_process(exam_pred, image_pred):
    
    rv_lv_ratio_lt_1_ix = CFG['exam_target_cols'].index('rv_lv_ratio_lt_1')
    rv_lv_ratio_gte_1_ix = CFG['exam_target_cols'].index('rv_lv_ratio_gte_1')
    central_pe_ix = CFG['exam_target_cols'].index('central_pe')
    rightsided_pe_ix = CFG['exam_target_cols'].index('rightsided_pe')
    leftsided_pe_ix = CFG['exam_target_cols'].index('leftsided_pe')
    acute_and_chronic_pe_ix = CFG['exam_target_cols'].index('acute_and_chronic_pe')
    chronic_pe_ix = CFG['exam_target_cols'].index('chronic_pe')
    negative_exam_for_pe_ix = CFG['exam_target_cols'].index('negative_exam_for_pe')
    indeterminate_ix = CFG['exam_target_cols'].index('indeterminate')
    
    # rule 1 or rule 2 judgement: if any pe image exist
    has_pe_image = torch.max(image_pred, 1)[0][0] > 0
    #print(has_pe_image)
    
    # rule 1-a: only one >= 0.5, the other < 0.5
    rv_lv_ratios = exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix]]
    rv_lv_ratios_1_a = nn.functional.softmax(rv_lv_ratios, dim=1) # to make one at least > 0.5
    rv_lv_ratios_1_a = torch.log(rv_lv_ratios_1_a/(1-rv_lv_ratios_1_a)) # turn back into logits
    exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix]] = torch.where(has_pe_image, rv_lv_ratios_1_a, rv_lv_ratios)
    
    # rule 1-b-1 or 1-b-2 judgement: at least one > 0.5
    crl_pe = exam_pred[:, [central_pe_ix, rightsided_pe_ix, leftsided_pe_ix]]
    has_no_pe = torch.max(crl_pe ,1)[0] <= 0 # all <= 0.5
    #print(has_no_pe)
    #assert False
        
    # rule 1-b
    max_val = torch.max(crl_pe, 1)[0]
    crl_pe_1_b = torch.where(crl_pe==max_val, 0.0001-crl_pe+crl_pe, crl_pe)
    exam_pred[:, [central_pe_ix, rightsided_pe_ix, leftsided_pe_ix]] = torch.where(has_pe_image*has_no_pe, crl_pe_1_b, crl_pe)
    
    # rule 1-c-1 or 1-c-2 judgement: at most one > 0.5
    ac_pe = exam_pred[:, [acute_and_chronic_pe_ix, chronic_pe_ix]]
    both_ac_ch = torch.min(ac_pe ,1)[0] > 0 # all > 0.5
    
    # rule 1-c
    ac_pe_1_c = nn.functional.softmax(ac_pe, dim=1) # to make only one > 0.5
    ac_pe_1_c = torch.log(ac_pe_1_c/(1-ac_pe_1_c)) # turn back into logits
    exam_pred[:, [acute_and_chronic_pe_ix, chronic_pe_ix]] = torch.where(has_pe_image*both_ac_ch, ac_pe_1_c, ac_pe)
    
    # rule 1-d
    neg_ind = exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]]
    neg_ind_1d = torch.clamp(neg_ind, max=0)
    exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]] = torch.where(has_pe_image, neg_ind_1d, neg_ind)
    
    # rule 2-a
    ne_inde = exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]]
    ne_inde_2_a = nn.functional.softmax(ne_inde, dim=1) # to make one at least > 0.5
    ne_inde_2_a = torch.log(ne_inde_2_a/(1-ne_inde_2_a)) # turn back into logits
    exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]] = torch.where(~has_pe_image, ne_inde_2_a, ne_inde)
    
    # rule 2-b
    all_other_exam_labels = exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix,
                                          central_pe_ix, rightsided_pe_ix, leftsided_pe_ix,
                                          acute_and_chronic_pe_ix, chronic_pe_ix]]
    all_other_exam_labels_2_b = torch.clamp(all_other_exam_labels, max=0)
    exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix,
                  central_pe_ix, rightsided_pe_ix, leftsided_pe_ix,
                  acute_and_chronic_pe_ix, chronic_pe_ix]] = torch.where(~has_pe_image, all_other_exam_labels_2_b, all_other_exam_labels)
    
    return exam_pred, image_pred
    
def check_label_consistency(checking_df):
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
    df = checking_df.copy()
    print(df.shape)
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())

    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]

    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 
                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)
    rule1a['broken_rule'] = '1a'

    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 
                        (df_pos.rightsided_pe <= 0.5) & 
                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)
    rule1b['broken_rule'] = '1b'

    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 
                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)
    rule1c['broken_rule'] = '1c'
    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS

    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'

    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 
                         (df_neg.negative_exam_for_pe >  0.5)) | 
                        ((df_neg.indeterminate        <= 0.5)  & 
                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)
    rule2a['broken_rule'] = '2a'

    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 
                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |
                        (df_neg.central_pe           > 0.5) | 
                        (df_neg.rightsided_pe        > 0.5) | 
                        (df_neg.leftsided_pe         > 0.5) |
                        (df_neg.acute_and_chronic_pe > 0.5) | 
                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)
    rule2b['broken_rule'] = '2b'
    # MERGING INCONSISTENT PREDICTIONS
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)
    
    print('label in-consistency counts:', errors.shape)
        
    if errors.shape[0] > 0:
        print(errors.broken_rule.value_counts())
        print(errors)
        assert False

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

def inference(model, device, df, root_path):
    model.eval()

    t = time.time()

    ds = RSNADataset(df, 0.0, root_path, STAGE1_CFGS=STAGE1_CFGS, image_subsampling=False, transforms=None, output_label=False) # change transforms=get_valiid_augmentation() to avoid TTA, or tta_augmentation()
    
    dataloader = torch.utils.data.DataLoader(
        ds, 
        batch_size=1,
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    
    patients = ds.get_patients()
    
    res_dfs = []

    for step, (imgs, per_image_preds, locs, img_num, index, seq_ix) in enumerate(dataloader):
        imgs = imgs.to(device).float()
        per_image_preds = per_image_preds.to(device).float()
        locs = locs.to(device).float()
        
        index = index.detach().numpy()[0]
        seq_ix = seq_ix.detach().numpy()[0,:]
        
        patient_filt = (df.StudyInstanceUID == patients[index])
        
        patient_df = pd.DataFrame()
        patient_df['SOPInstanceUID'] = df.loc[patient_filt, 'SOPInstanceUID'].values[seq_ix]
        patient_df['SeriesInstanceUID'] = df.loc[patient_filt, 'SeriesInstanceUID'].values # no need to sort
        patient_df['StudyInstanceUID'] = patients[index] # single value
        
        for c in CFG['image_target_cols']+CFG['exam_target_cols']:
            patient_df[c] = 0.0

        #with autocast():
        image_preds, exam_pred = model(per_image_preds, locs)   #output = model(input)
        
        exam_pred = torch.sigmoid(exam_pred).cpu().detach().numpy()
        image_preds = torch.sigmoid(image_preds).cpu().detach().numpy()

        patient_df[CFG['exam_target_cols']] = exam_pred[0]
        patient_df[CFG['image_target_cols']] = image_preds[0,:]
        res_dfs += [patient_df]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(dataloader)):
            print(
                f'Inference Step {step+1}/{len(dataloader)}, ' + \
                f'time: {(time.time() - t):.4f}', end='\r' if (step + 1) != len(dataloader) else '\n'
            )
    
    res_dfs = pd.concat(res_dfs, axis=0).reset_index(drop=True)
    res_dfs = df[['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID']].merge(res_dfs, on=['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID'], how='left')
    print(res_dfs[CFG['image_target_cols']+CFG['exam_target_cols']].head(5))
    print(res_dfs[CFG['image_target_cols']+CFG['exam_target_cols']].tail(5))
    assert res_dfs.shape[0] == df.shape[0]
    
    return res_dfs

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

def get_stage1_columns():
    
    new_feats = []
    for cfg in STAGE1_CFGS:
        for i in range(cfg['output_len']):
            f = cfg['tag']+'_'+str(i)
            new_feats += [f]
        
    return new_feats

def update_stage1_test_preds(df):
    
    new_feats = get_stage1_columns()
    for f in new_feats:
        df[f] = 0

        
    test_ds = RSNADatasetStage1(df, 0.0, CFG['test_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=False) # transforms=get_valid_transforms() or transforms=tta_augmentation()
        
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=256,
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
        sampler=SequentialSampler(test_ds)
    )
    image_preds_all_list = []
    models = []
    
    for cfg in STAGE1_CFGS:
        device = torch.device(CFG['device'])
        model = cfg['model_constructor']().to(device)
        model.load_state_dict(torch.load('{}/model_{}'.format(CFG['model_path'], cfg['tag'])))
        model.eval()
        models.append(model)
    
    image_preds_all = []
    for step, imgs in enumerate(tqdm(test_loader)):
        imgs = torch.reshape(imgs, (-1, 3, 256, 256))
        imgs = imgs.to(device).float()
        for model in models:
            image_preds = model(imgs)   #output = model(input)
            image_preds_all += [image_preds.cpu().detach().numpy()]
        #print(imgs[0], image_preds[0,:]); break
        #continue
    del models, test_loader

    torch.cuda.empty_cache()

    image_preds_all_image = np.concatenate(image_preds_all[::2], axis=0)
    image_preds_all_exam = np.concatenate(image_preds_all[1::2], axis=0)
    
    image_preds_all = np.concatenate([image_preds_all_image, image_preds_all_exam], axis=1)

    #image_preds_all = np.concatenate(image_preds_all, axis=1)
    print(np.array(new_feats).shape)
    print(np.array(image_preds_all).shape)
    df.loc[:,new_feats] = image_preds_all

    return df

if __name__ == '__main__':
    with open('config.json') as json_file: 
        CFG = json.load(json_file) 

    seed_everything(SEED)
    
    from os import path
    if path.exists('../input/rsna-str-pulmonary-embolism-detection/train') and not do_full:
        test_df = pd.read_csv(CFG['test_path']).head(1000)
    else:
        test_df = pd.read_csv(CFG['test_path'])

    with torch.no_grad():
        test_df = update_stage1_test_preds(test_df)

    device = torch.device(CFG['device'])
    model = RSNAClassifier(STAGE1_CFGS=STAGE1_CFGS).to(device)
    model.load_state_dict(torch.load('{}/model_{}'.format(CFG['model_path'], CFG['tag'])))
    test_pred_df = inference(model, device, test_df, CFG['test_img_path'])       
    test_pred_df.to_csv('submission_raw.csv')

    # transform into submission format
    ids = []
    labels = []

    gp_mean = test_pred_df.loc[:, ['StudyInstanceUID']+CFG['exam_target_cols']].groupby('StudyInstanceUID', sort=False).mean()
    for col in CFG['exam_target_cols']:
        ids += [[patient+'_'+col for patient in gp_mean.index]]
        labels += [gp_mean[col].values]

    ids += [test_pred_df.SOPInstanceUID.values]
    labels += [test_pred_df[CFG['image_target_cols']].values[:,0]]
    ids = np.concatenate(ids)
    labels = np.concatenate(labels)

    assert len(ids) == len(labels)

    submission = pd.DataFrame()
    submission['id'] = ids
    submission['label'] = labels
    print(submission.head(3))
    print(submission.tail(3))
    print(submission.shape)
    submission.to_csv('submission.csv', index=False)
