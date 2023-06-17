import numpy as np
import sys
sys.path.insert(0, '/home/eddieh00/UCSD/ms/ece228/bmi_model/facenet/src')
import facenet
import tensorflow as tf
import os
from scipy import misc
from skimage.transform import resize
from align import detect_face
import imageio
import cv2
#########

def load_and_align_image(file_path, image_size=160, margin=32, gpu_memory_fraction=1.0):
    # Create a list of input images
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Read image
    img = imageio.imread(os.path.expanduser(file_path))

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
  
    # Detect face in the image
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    
    # If no face is detected, return original image
    if bounding_boxes.shape[0] == 0:
        print("No face detected")
        return img

    # Assuming the image has only one face, get the bounding box
    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin//2, 0)
    bb[1] = np.maximum(det[1]-margin//2, 0)
    bb[2] = np.minimum(det[2]+margin//2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin//2, img.shape[0])

    # Crop the image using the bounding box coordinates
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

    # Resize the cropped image to the desired size
    aligned = resize(cropped, (image_size, image_size), mode='reflect')
    aligned = (aligned * 255).astype(np.uint8)

    return aligned