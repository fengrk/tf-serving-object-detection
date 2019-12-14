# coding:utf-8
__author__ = 'rk.feng'
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from grpc.beta import implementations
from tensorboard._vendor.tensorflow_serving.apis import prediction_service_pb2, predict_pb2

_cur_dir = os.path.dirname(__file__)


def load_image_into_numpy_array(_image):
    (im_width, im_height) = _image.size
    return np.array(_image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def call_tf_serving(image_list: [str], server: str):
    """ """
    # Create stub
    if len(image_list):
        _image = load_image_into_numpy_array(Image.open(image_list[0]))
        image_np = np.zeros(shape=(len(image_list), _image.shape[0], _image.shape[1], 3), dtype=np.uint8)
        image_np[0] = _image
        for i in range(len(image_list) - 1):
            image_np[i + 1] = load_image_into_numpy_array(Image.open(image_list[i + 1]))
    else:
        image_np = load_image_into_numpy_array(Image.open(image_list[0]))
        image_np = np.reshape(image_np, newshape=[1] + list(image_np.shape))

    host, port = server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Create prediction request object
    request = predict_pb2.PredictRequest()

    # Specify model name (must be the same as when the TensorFlow serving serving was started)
    request.model_spec.name = 'object'

    # Initalize prediction
    # Specify signature name (should be the same as specified when exporting model)
    request.model_spec.signature_name = "serving_default"
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_np, shape=image_np.shape))

    # Call the prediction server
    _result = stub.Predict(request, 20.0 * len(image_list))  # 20 secs timeout

    #
    _dict = {
        "detection_boxes": np.reshape(_result.outputs['detection_boxes'].float_val, newshape=(-1, 4)),
        "detection_classes": np.squeeze(_result.outputs['detection_classes'].float_val).astype(np.int32),
        "detection_scores": np.squeeze(_result.outputs['detection_scores'].float_val),
    }
    length = len(image_np)
    for key, value in _dict.items():
        _new_shape = list(value.shape)
        _new_shape.insert(0, -1)
        _new_shape[1] = _new_shape[1] // length
        _dict[key] = np.reshape(value, newshape=_new_shape)

    output_dict_list = []
    for i in range(length):
        output_dict_list.append({
            "detection_boxes": _dict['detection_boxes'][i],
            "detection_classes": _dict['detection_classes'][i],
            "detection_scores": _dict['detection_scores'][i],
        })

    return output_dict_list


if __name__ == '__main__':
    print(
        call_tf_serving(
            image_list=[
                os.path.join(_cur_dir, "test/dog.jpg"),
            ],
            server="127.0.0.1:8500"
        )
    )
