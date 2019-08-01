import os
# import sys
# import tarfile
# import zipfile
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
# import six.moves.urllib as urllib
import centroidtracker

# from io import StringIO
# from collections import defaultdict
from distutils.version import StrictVersion
# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
# from PIL import Image
# from models.research.object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
# print(os.getcwd())
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
matplotlib.use('Agg')
NUM_CLASSES = 90

cam_num = 1
cam_way = ''

if cam_num is 1:
    cam_way = 'rtsp://admin:DMD2010@192.168.50.68/live1.sdp'  # out 308
    # protocol://user:password@IP:port
elif cam_num is 2:
    cam_way = 'rtsp://admin:DMD2010@192.168.50.72/live1.sdp'  # in 308
elif cam_num is 3:
    cam_way = 'rtsp://admin:DMD2010!@@dmd-axxon:566/1'  # out 308, axxon
elif cam_num is 4:
    cam_way = 'rtsp://admin:DMD2010@dmd-axxon:566/2'  # in 308, axxon
else:
    cam_way = 'test.avi'  # out 308

cap = cv2.VideoCapture(cam_way)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
# \Users\vyach\PycharmProjects\TensorFlowTest\models\research\object_detection
# Users/vyach/PycharmProjects/TensorFlowTest/models/research/object_detection/data
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('models/research/object_detection/data', 'mscoco_label_map.pbtxt')

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())

# PATH_TO_TEST_IMAGES_DIR = 'models\\research\object_detection\\images'
# #TEST_IMAGE_PATHS = 'models\research\object_detection\test_images\image1.jpg'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(
    os.path.join('models/research/object_detection/data', 'mscoco_label_map.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
min_confidence = 0.7


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    # im_height = np.size(image, 0)
    # im_width = np.size(image, 1)
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


with detection_graph.as_default():
    maxDis = 7
    ct = centroidtracker.CentroidTracker(maxDis)
    with tf.Session() as sess:
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        while True:
            ret, image_np = cap.read()
            # image_np = cv2.resize(image_np, (800, 600))
            image_np = cv2.resize(image_np, (1024, 512))
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.

            (h, w) = image_np.shape[:2]

            person_i_list = list()
            i = 0
            for x in output_dict["detection_classes"]:
                if x == 1:
                    # if x is person
                    if output_dict['detection_scores'][i] >= min_confidence:
                        person_i_list.append(i)
                i += 1

            personal_boxes = []
            person_images = []
            person_images_p = []
            if person_i_list.__len__() is not 0:
                for x in person_i_list:
                    box = output_dict['detection_boxes'][x]
                    # need to reverse [minY minX maxY maxX] to [minX minY maxX maxY]
                    rbox = np.array([box[1], box[0], box[3], box[2]])
                    # denormalize coordinates
                    box = rbox * np.array([w, h, w, h])
                    box = box.astype("int")
                    personal_boxes.append(box)

                    # (startX, startY, endX, endY) = rects[i]
                    # sub_f = frame[startY:endY, startX:endX]
                    person_images.append(image_np[box[1]:box[3], box[0]:box[2]])
                    person_images_p.append(output_dict['detection_scores'][x])

            # Not so simple, needs list of images!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            objects = ct.update(personal_boxes, person_images, person_images_p)
            '''
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            '''
            for x in personal_boxes:
                # reversed X and Y
                (startX, startY, endX, endY) = x.astype("int")
                cv2.rectangle(image_np, (startX, startY), (endX, endY), (30, 30, 30), 2)

            for (objectID, centroid) in objects.items():
                X = centroid[0]
                Y = centroid[1]

                text = "ID {}".format(objectID)
                cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image_np, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                # x, y
                # print("ID: " + objectID.__str__() +
                #      " ;centr: X: " + X.__str__() + "; Y: " + Y.__str__())
            cv2.imshow('object detection', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
