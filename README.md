# Door Detection Model
This repository trains a model to detect the bounding boxes of doors present in PNG images of CAD-based indoor floorplans. It uses floorplan data from the CubiCasa5k dataset to perform training and evaluation. All data downloading, preprocessing, training and evaluation is performed as part of the [provided notebook](https://github.com/morshed-research/Door_Detection_Model/blob/main/fine_tune.ipynb) in this repository. 

## Installing Dependencies
Requires Python 3.10.12

Run:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
