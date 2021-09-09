#Path Defining
main_dir_path = 'D:/Codes/Python/3D_printer_Object_Detection/Tensorflow'
worspace_path = main_dir_path + '/workspace'
scripts_path = main_dir_path + '/scripts'
Apimodels_path = main_dir_path + '/models'
annotations_path = worspace_path + '/annotations'
image_path = worspace_path + '/images'
model_path = worspace_path + '/models'
pre_trained_model_path = worspace_path + '/pre-trained-models'
config_path = model_path + '/my_ssd_mobnet/pipeline.config'
checkpoint_path = model_path + '/my_ssd_mobne'

#setting labels

labels = [{'name':'Good','id':1},{'name':'Fail','id':2}]

''' class directory(object):
    def __init__(self,given):
        self.path = given
    def __enter__(self):
        self.labelmap = open(self.path,'w') 
        return self.labelmap
    def __exit__(self):
        self.labelmap.close() '''

#label map for final values in Deep learning

with open(annotations_path+'/label_map.pbtxt','w') as labelmap :
    for label in labels:
        labelmap.write("item {\n")
        labelmap.write("\tname:\'{}\'\n".format(label['name']))
        labelmap.write("\tid:{}\n".format(label['id']))
        labelmap.write('}\n')

#importing Libraries

import os

#creating tf.records

os.system('python3 {}+/generate_tfrecord.py -x {}+/train -l {}+/label_map.pbtxt -o {}+/train.record'.format(scripts_path,image_path,annotations_path,annotations_path))
os.system('python3 {}+/generate_tfrecord.py -x {}+/test -l {}+/label_map.pbtxt -o {}+/test.record'.format(scripts_path,image_path,annotations_path,annotations_path))

#downloading api model

os.system('cd {} && git clone https://github.com/tensorflow/models'.format(main_dir_path))

#importing pipeline to output folder

custom_model_name = 'my_ssd_mobne'
os.system('cd {} && mkdir {}'.format(model_path,custom_model_name))
os.system('cp {}/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config {}/{}'.format(pre_trained_model_path,model_path,custom_model_name))

#updating config for transfer learning

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config_path = model_path + '/' + custom_model_name + '/pipeline.config'
config = config_util.get_configs_from_pipeline_file(config_path)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(config_path,'r') as f:               #??
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
    
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(config_path,'wb') as f:            #??
    f.write(config_text)

#training model according to our tfrecords

''' os.system('python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={} --num_train_steps=20400'.format(Apimodels_path,model_path,custom_model_name,config_path)) '''

import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

detection_model = model_builder.build(model_config=config['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(checkpoint_path,'ckpt-21').replace('\\','/')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image,shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

import cv2 as cv
import numpy as np

category_index = label_map_util.create_category_index_from_labelmap(annotations_path +'/label_map.pbtxt')

cap = cv.VideoCapture(0)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np,0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detetctions = int(detections.pop('num_detections'))
    detections = {key:value[0, :num_detetctions].np() for key,value in detections.items()}
    detections['num_detections'] = num_detetctions

    detections['detection_classes'] = detections['detections_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5,
        agnostic_mode=False)
    cv.imshow('3D Printer AI Detection',cv.resize(image_np_with_detections,(800,600)))

    if cv.waitkey(1) & 0xFF == ord('q'):
        cap.release()
        break
