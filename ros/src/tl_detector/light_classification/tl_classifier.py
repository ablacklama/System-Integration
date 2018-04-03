#Class to handle all the traffic light detection for Carlai
import numpy as np
import rospy
import tensorflow as tf
from tensorflow.python.client import device_lib
from styx_msgs.msg import TrafficLight

#class is made to have one instance of it created and used for all images
class TLClassifier(object):
    def __init__(self, is_simulator=True, cpu=False):
        """
		model_dir = path to directory containing 'frozen_inference_graph.pb'
		cpu = False if you want the model to run on gpu.
		cpu should only be true if you don't have enough gpu memory to contain the model.
		"""

        use_gpu = False

        # System has GPU?
        gpu_aval = [x for x in device_lib.list_local_devices() if x.device_type
                == 'GPU']
        if (not cpu) and gpu_aval:
            use_gpu = True
        else:
            use_gpu = False

        # Default model path is for the simulator
        if is_simulator:
            PATH_TO_MODEL = 'light_classification/frozen_inference_graph.pb'
        else:
            PATH_TO_MODEL = 'light_classification/real_frozen_inference_graph.pb'

        self.model_graph = tf.Graph()

        #if os.path.isfile(PATH_TO_MODEL):
        #    rospy.logerr("Traffic light model file deos not exist; {0}".
        #            format(PATH_TO_MODEL))
        with self.model_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
            self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
            self.t_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
            self.t_scores = self.model_graph.get_tensor_by_name('detection_scores:0')
            self.t_classes = self.model_graph.get_tensor_by_name('detection_classes:0')

        if not use_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(graph=self.model_graph, config=config)
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.model_graph, config=config)

    def predict(self, img):
        """
		Preform inference on one RGB image array.
		Returns the unfiltered bboxes, scores, and accociated classes
		"""
        with self.model_graph.as_default():
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes) = self.sess.run([self.t_boxes, self.t_scores, self.t_classes],
                                                     feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes

    def filter_boxes(self, min_score, boxes, scores, classes):
        """
        Takes the bboxes, scores and classes from self.predict()
        Return boxes with a confidence >= `min_score`
        """
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs]
        filtered_scores = scores[idxs]
        filtered_classes = classes[idxs]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The SSD boxes are normalized to between 0 and 1.
        This converts them to actual pixel locations on the image
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def pipeline(self, img, confidence_cutoff=.75):
        """
    	Take an RGB image and predict class and score arrays for traffic lights.
    	Confidence cutoff determains how how sure
    	the model has to be for us to accept a light as detected.

    	Classes:
    	1 = green
    	2 = yellow
    	3 = red
        """
        boxes, scores, classes = self.predict(img)
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        return classes, scores

    def visual_pipeline(self, img, confidence_cutoff=.75):
        """
    	does everything in pipeline
    	also returns the image and pixel coordinates of the box for testing
        """

        boxes, scores, classes = self.predict(img)

        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        h, w, _ = img.shape
        box_coords = self.to_image_coords(boxes, h, w)

        return classes, scores, img, box_coords

    def get_classification(self, image, confi_thresh=0.75):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        classes, scores = self.pipeline(image, confidence_cutoff=confi_thresh)

        if len(classes) == 0:
            return TrafficLight.UNKNOWN, 0
        else:
            if classes[0] == 1:
                return TrafficLight.GREEN, scores[0]
            elif classes[0] == 2:
                return TrafficLight.YELLOW, scores[0]
            elif classes[0] == 3:
                return TrafficLight.RED, scores[0]
