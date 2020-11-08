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

Todo:
- Stage 2 GRU training script
- Inference script
- Add the ability to change models (from b0 to bx and add ResNeSt)
- Postprocessing/label consistency check
- Docstrings

Any files in the `Kaggle` directory are verified to work on Kaggle.
Shared scripts and instructions to run are inside of the readme in the directory.

JSON must be called config.json and in current directory.

Stratified Validation Strategy from [Kun's Notebook](https://www.kaggle.com/khyeh0719/stratified-validation-strategy).