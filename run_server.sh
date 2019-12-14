#!/bin/bash

# download object detection model
pretrained_model_dir="faster_rcnn_resnet50_coco_2018_01_28"
pretrained_model_url="http://download.tensorflow.org/models/object_detection/${pretrained_model_dir}.tar.gz"
ls model.tar.gz || (wget -c  -o model.tar.gz "${pretrained_model_url}" && mkdir -p pretrained && tar -xzf model.tar.gz --directory ./pretrained/)

# create saved model
PYTHONPATH=$(pwd) python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path="$(pwd)/pretrained/${pretrained_model_dir}/pipeline.config" \
    --output_directory="$(pwd)/server/object_model" \
    --trained_checkpoint_prefix="$(pwd)/pretrained/${pretrained_model_dir}/model.ckpt" \
    --saved_model_with_variables=True
mv server/object_model/saved_model server/object_model/1

# run tf server
cd server && docker-compose up -d

