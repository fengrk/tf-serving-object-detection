#!/bin/bash

# download object detection model
pretrained_model_url="http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz"
ls model.tar.gz || (wget -c  -o model.tar.gz "${pretrained_model_url}" && mkdir -p pretrained && tar -xzf model.tar.gz --directory ./pretrained/)

# create saved model
python build_saved_model.py `pwd`/pretrained/

# run tf server
cd server && docker-compose up -d


