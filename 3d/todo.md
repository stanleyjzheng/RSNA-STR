# Steps for 3d augmentations
### STILL DOES NOT WORK!
1. Redo the stage 1 dataloader in `utils.py` so that it returns one study of images augmented using volumentations as well as the ordered labels. The easiest way to do this would be to group all of the images by study, then pass the study identifier to the dataloader. Then, the dataloader retrieves paths for all of the studies (may be faster to do this once statically) and pass the paths to `get_img()` to retrieve the images themselves.
This is done but I'm not sure how well it works.
2. Modify `update_stage1_oof_preds()` to iteratively predict and save to a CSV in the correct format. This is likely going to be a bottleneck so optimization here is crucial.
3. Somehow modify validation to do the same thing efficiently

Sounds easy, but OOM is such a massive problem. Going to have to do this smart. 