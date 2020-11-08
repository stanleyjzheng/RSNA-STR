# RSNA-STR Code

Done so far:
- Stage 1 multilabel training script
- Stage 1 image level training script

Todo:
- Stage 2 GRU training
- Inference
- Add the ability to change models (from b0 to bx and add ResNeSt)
- Postprocessing/label consistency
- Docstrings

Any files in the `Kaggle` directory are verified to work on Kaggle.
Shared scripts and instructions to run are inside of the readme in the directory.

JSON must be called config.json and in current directory.

Stratified Validation Strategy from [Kun's Notebook](https://www.kaggle.com/khyeh0719/stratified-validation-strategy).