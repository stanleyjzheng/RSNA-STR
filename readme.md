# RSNA-STR Code

### About
I created this with the intention to centralize all code and make it more maintainable.
Docstrings have been written where possible, and code is more organized and systematic than the Kaggle notebooks.

### Progress
Done so far:
- Stage 1 multilabel training script ([Shared Kaggle Notebook](https://www.kaggle.com/stanleyjzheng/rsna-github-multilabel-testing?scriptVersionId=468178756))
- Stage 1 image level training script ([Shared Kaggle Notebook](https://www.kaggle.com/stanleyjzheng/rsna-github-image-level-testing?scriptVersionId=46817626))
- Stage 2 GRU training script ([Shared Kaggle Notebook](https://www.kaggle.com/stanleyjzheng/rsna-str-stage-2-gru-train?scriptVersionId=46890528))

Todo in order of importance:
- Make Stage 2 GRU training more representative of real inference (add train transforms, improve `RSNADataset`)
- 3d augmentations (currently committed but not working and extremely experimental)
- Docstrings

### About `config.json`
Parameters to be changed:
- img_size: Image size (square)
- lr: Learning rate
- epochs
- train_bs/valid_bs: training/validation batch size
- save_path: Model save path
- verbose_step: Number of steps between printing metrics
- num_workers: Number of threads to run concurrent processes with
- efbnet: Which efficientnet architecture to use. For example, 'efbnet': 'efficientnet-b7'
- train_folds: Nested list with folds to train with. Dimension 0 is the number of folds to run.

### Using on Discovery Cluster
Please run `./run.sh` to install dependencies. 
Note that this is untested, and I have no clue how to download the 910gb dataset.
`dataset.sh` contains a small script to clone the Kaggle RSNA dataset into `input` as well as a few sanity check models.
Then, simply run each script with `python3 script_to_run.py` 

### Using on Kaggle
To use these scripts on Kaggle, simply add the following datasets:
- [RSNA-STR Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data)
- [EfficientNet Pytorch 0.7](https://www.kaggle.com/tunguz/efficientnet-pytorch-07)
- [gdcm conda install](https://www.kaggle.com/ronaldokun/gdcm-conda-install)
- [RSNA STR GitHub (a clone of this repo)](www.kaggle.com/dataset/f4127c3bf3b0b540d8d17e1b4f1bddbe4ea05231c9613619e8ccd745c7dd2b17)

Then, the following lines can be added to the beginning of a script or notebook. 
The config file will be created during each run (since updating `config.json` requires a new Kaggle dataset).

```python
package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
import sys; sys.path.append(package_path); sys.path.append('./')

bash_commands = [
            'cp ../input/gdcm-conda-install/gdcm.tar .',
            'tar -xvzf gdcm.tar',
            'conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2',
            'cp ../input/rsna-str-github/utils.py .'
            ]

import subprocess
for bashCommand in bash_commands:
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

CFG = {
    'train': True,
    'train_img_path': '../input/rsna-str-pulmonary-embolism-detection/train',
    'test_img_path': '../input/rsna-str-pulmonary-embolism-detection/test',
    'cv_fold_path': '../input/rsna-str-github/rsna_train_splits_fold_20.csv',
    'train_path': '../input/rsna-str-pulmonary-embolism-detection/train.csv',
    'test_path': '../input/rsna-str-pulmonary-embolism-detection/test.csv',
    'image_target_cols': [
        'pe_present_on_image', # only image level
        'rv_lv_ratio_gte_1', # exam level
        'rv_lv_ratio_lt_1', # exam level
        'leftsided_pe', # exam level
        'chronic_pe', # exam level
        'rightsided_pe', # exam level
        'acute_and_chronic_pe', # exam level
        'central_pe', # exam level
        'indeterminate' # exam level
    ],
    'img_num': 200,
    'img_size': 256,
    'lr': 0.0005,
    'epochs': 1,
    'device': 'cuda', # cuda, cpu
    'train_bs': 64,
    'valid_bs': 256,
    'accum_iter': 1,
    'verbose_step': 1,
    'num_workers': 4,
    'efbnet': 'efficientnet-b0',
    
    'train_folds': [list(range(0, 16))],
    
    'valid_folds': [list(range(17, 21))],
    
    'model_path': '../input/kh-rsna-model',
    'save_path': '.',
    'tag': 'efb0_stage1_multilabel',
}
import json
with open('config.json', 'w+') as outfile:
    json.dump(CFG, outfile, indent=4)
```