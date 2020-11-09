# RSNA-STR Code

### About
I created this with the intention to centralize all code and make it more maintainable.
Docstrings have been written where possible, and code is more organized and systematic than the Kaggle notebooks.

All static or commonly used code is in `utils.py`, including all datasets, dicom reading functions, etc.

`stage1imagelevel.py` contains the training loop for image level targets, based on [Kun's corresponding notebook called "CNN-GRU Baseline- Stage2 Train+Inference"](https://www.kaggle.com/khyeh0719/cnn-gru-baseline-stage2-train-inference)

`stage1examlevel.py` contains the training loop for exam level targets, based on [Kun's corresponding notebook called "CNN- Stage1 Train"](https://www.kaggle.com/khyeh0719/cnn-stage1-train)

### Progress
Done so far:
- Stage 1 multilabel training script
- Stage 1 image level training script
- TTA and non-TTA inference scripts (untested, please make sure 'train': False in config)
- - TTAx3 inference runs in 8.5hrs with EfficientnetB0
- - Non-TTA inference runs in 2.5 hrs with EfficientnetB0

Todo:
- Fix the postprocessing function in inference
- Stage 2 GRU training script
- Postprocessing/label consistency check
- Docstrings

Any files in the `Kaggle` directory are verified to work on Kaggle.
Shared scripts and instructions to run are inside of the readme in the directory.

### About `config.json`
JSON must be called config.json and in the same directory as `utils.py` and the training script.

Parameters:
- train: Set to true if training, false if doing inference
- train_img_path: Path to directory containing train images. On Kaggle, defaults to `../input/rsna-str-pulmonary-embolism-detection/train`
- test_img_path: Path to directory containing test images. On Kaggle, defaults to `../input/rsna-str-pulmonary-embolism-detection/test`
- cv_fold_path: Path to CSV containing study folds. Can be downloaded [here](https://www.kaggle.com/khyeh0719/stratified-validation-strategy)
- train_path: Path to `train.csv`
- test_path: Path to `test.csv`
- image_target_cols: List of target columns including image level
- img_size: Image dimension (square)
- lr: Learning rate
- accum_iter: Accumulative iteration (set to same as epochs)
- verbose_step: Number of steps between printing metrics
- num_workers: Number of threads to run concurrent processes with
- efbnet: Which efficientnet architecture to use. For example, 'efbnet': 'efficientnet-b7'
- train_folds: Nested list with folds to train with. Dimension 0 is the number of folds to run.

Stratified Validation Strategy from [Kun's Notebook](https://www.kaggle.com/khyeh0719/stratified-validation-strategy).