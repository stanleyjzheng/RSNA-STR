# RSNA-STR Code

Done so far:
- Stage 1 multilabel script

Todo:
- Stage 1 image level script
- Stage 2 GRU training
- Inference
- Add the ability to change models
- Postprocessing/label consistency
- Docstrings
- Clarify dataset names

Currently very experimental and unstable code. I will do my best to make this code more stable and usable, as I believe this format is much easier to work with than repetitive notebooks in the long term. For now, I am still experimenting with JSON config and splitting everything into scripts.

Any files in the `Kaggle` directory are verified to work on Kaggle and notebooks and instructions to run are shared.

JSON must be called config.json and in current directory. (change  to input?)

Stratified Validation Strategy from [Kun's Notebook](https://www.kaggle.com/khyeh0719/stratified-validation-strategy).