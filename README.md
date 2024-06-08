# Door Detection Model
This repository trains a model to detect the bounding boxes of doors present in PNG images of CAD-based indoor floorplans. It uses floorplan data from the CubiCasa5k dataset to perform training and evaluation. All data downloading, preprocessing, training and evaluation is performed as part of the [provided notebook](https://github.com/morshed-research/Door_Detection_Model/blob/main/fine_tune.ipynb) in this repository. 

## Installing Dependencies
Requires Python 3.10.12

Run:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Extracting Door Bounding Boxes
From the CubiCasa5k floor plan SVGs, the bounding box coordinates of doors are extracted as follows:
- First the min and max x,y coordinates of the door's entry rectangle threshold are taken:
   - Find the polygon tag enclosed within the outer door tag
   - Access its point attribute and split up its coordinate set
- Next we find the coordinate marking the end of the door's curve:
  - Find the path tag enclosed within the outer door tag
  - Remove all flag indicators of movement direction
  - Sum the resulting x and y coordinates respectively
- Add each found coordinate pair to a list of x and list of y coordinates
- Take the smallest x and y coordinate for the start of the bounding box and the largest x and y coordinates for the end of the bounding box
