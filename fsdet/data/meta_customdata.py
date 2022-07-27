import os
import xml.etree.ElementTree as ET

import numpy as np
import glob
import json
from labelme import utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def extract_info_from_json(ann_path, count, thing_classes):
    img_ppt = {}
    with open(ann_path, 'r') as fp:
        data = json.load(fp)
        #get path to original image
        img_ppt['file_name'] = data['imagePath']
        img_ppt['image_id'] = count
        img_ppt['height'], img_ppt['width'] = utils.img_b64_to_arr(data["imageData"]).shape[:2]
        img_ppt['annotations'] = []
        #get annotation coordinates and labels
        for shapes in data['shapes']:
            annotation = {}
            annotation['category_id'] = thing_classes.index(shapes['label'])
            points = shapes['points']
            x0, y0, width, height = pointsTobbox(points)
            annotation['bbox'] = [x0, y0, width, height]
            annotation['bbox_mode'] = BoxMode.XYWH_ABS
            img_ppt['annotations'].append(annotation)
        return img_ppt
                    

def pointsTobbox(points):
    x_min,y_min = points[0][:]
    x_max,y_max = points[1][:]  
    x0, y0 = x_min, y_max
    width, height = x_max - x_min, y_max - y_min      
    return x0, y0, width, height

def load_custom_labelme_instances(dataset_name, thing_classes, dataset_dir):
    data = []
    for idx, ann_path in enumerate(glob.glob(os.path.join(dataset_dir, '*.json'))):
        data.append(extract_info_from_json(ann_path, idx, thing_classes))
    return data