# A class implementing a keypoint-matching object detector
# for the BelgaLogos dataset.
import lib.keypoint_matching as kpm
from collections import namedtuple
import cv2

# A container for detected objects
DetectedObject = namedtuple("DetectedObject", ["label", "bounding_box"])


def annotate_image_with_objects(image, detected_objects, correct_match=None):
    """
        For an input image and a list of DetectedObject tuples,
        returns a new image annotating the image with the detection
        bounding-boxes.

        `correct_match` is an optional input, consisting of a list of bools,
        the same length as the number of detected objects. If an element is
        True, then that bounding-box is rendered in green, if False it is
        rendered in red.
    """
    if correct_match == None:
        correct_match = [True] * len(detected_objects)

    colours = { True: (0, 255, 0),
                False: (0, 0, 255)}

    annotated_image = image.copy()
    for i, iobject in enumerate(detected_objects):
        box_colour = colours[correct_match[i]]
        cv2.polylines(annotated_image, [iobject.bounding_box], True, box_colour, 3, cv2.LINE_AA)
    return annotated_image


class KeypointMatcher:
    def __init__(self, finder, norm):
        self.finder = finder
        self.norm   = norm
        self.matcher = cv2.BFMatcher(norm, crossCheck=True)
        # Available categories
        self.templates = []
        self.labels = []

    def add_template(self, label, image):
        # Find template keypoints
        kp, desc = self.finder.detectAndCompute(image, None)
        template = kpm.KeypointSet(image, kp, desc)
        self.templates.append(template)
        self.labels.append(label)
        # Find template inverse keypoints
        inverse_image = cv2.bitwise_not(image)
        kp, desc = self.finder.detectAndCompute(inverse_image, None)
        template = kpm.KeypointSet(inverse_image, kp, desc)
        self.templates.append(template)
        self.labels.append(label)

    def detect_objects(self, target_image):
        kp_clusters, ds_clusters = kpm.meanshift_keypoint_clusters(target_image, self.finder)
        n_clusters = len(kp_clusters)
        n_labels = len(self.labels)
        detected_objects = []
        for ic in range(n_clusters):
            for il in range(n_labels):
                template = self.templates[il]
                label    = self.labels[il]
                cluster = kpm.KeypointSet(target_image, kp_clusters[ic], ds_clusters[ic])
                bounding_box = kpm.get_matching_boundingbox(template, cluster, self.matcher,
                                                               MIN_MATCHES=10, MIN_INLIERS=10)
                if bounding_box is not None:
                    new_object = DetectedObject(label, bounding_box)
                    detected_objects.append(new_object)
                    break
        return detected_objects
