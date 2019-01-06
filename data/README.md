# Cityscapes
1. Get dataset files: `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` from [here](https://www.cityscapes-dataset.com/downloads/).
2. Get [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) and do `cd cityscapesScripts; pip install .`.
3. Generate images with labels (`annotations` folder must contain `gtFine`):
```
export CITYSCAPES_DATASET=/home/dan/datasets/cityscapes/annotations/
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
4. Run `create_tfrecords_for_cityscapes.py`.

# ModaNet
[ModaNet: A Large-Scale Street Fashion Dataset with Polygon Annotations](https://arxiv.org/abs/1807.01394).


Files `train_metadata.csv` and `val_metadata.csv` contain
1. `id` of an image (this ids are used in json annotation files).
2. `path` - download path of an image.
3. `is_ok` - whether I was able to download this particular image.

For more information see the original dataset repository:  
[eBay/modanet](https://github.com/eBay/modanet)


## How to get ModaNet dataset
1. Go to [eBay/modanet](https://github.com/eBay/modanet) and download two annotation json files.
2. Install [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi).
3. Download Chictopia metadata tables (.sql.gz file) from [here](https://github.com/kyamagu/paperdoll/tree/master/data/chictopia).
4. The table `"photos"` contains ids and download paths of images.
5. Download what you can.

## How to create a train-val split
Annotations are only available for the official train part.  
So we need to create a train-val split by ourselves.    
Use `explore_modanet.ipynb` to explore random dataset annotations.  
Also it creates an label integer encoding for training (`modanet_labels.txt`).


train 46891 = 42243 +
val 2591
