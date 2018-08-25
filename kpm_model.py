# A class implementing a keypoint-matching object detector
# for the BelgaLogos dataset.

import keypoint_matching as kp_lib
import cv2


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
        template = kp_lib.KeypointSet(image, kp, desc)
        self.templates.append(template)
        self.labels.append(label)
        # Find template inverse keypoints
        inverse_image = cv2.bitwise_not(image)
        kp, desc = self.finder.detectAndCompute(inverse_image, None)
        template = kp_lib.KeypointSet(inverse_image, kp, desc)
        self.templates.append(template)
        self.labels.append(label)

    def matching_bounding_boxes(self, target_image):
        kp_clusters, ds_clusters = kp_lib.meanshift_keypoint_clusters(target_image, self.finder)
        n_clusters = len(kp_clusters)
        n_labels = len(self.labels)
        match_bounding_boxes = []
        match_labels = []
        for ic in range(n_clusters):
            for il in range(n_labels):
                template = self.templates[il]
                label    = self.labels[il]
                cluster = kp_lib.KeypointSet(target_image, kp_clusters[ic], ds_clusters[ic])
                bounding_box = kp_lib.get_matching_boundingbox(template, cluster, self.matcher,
                                                               MIN_MATCHES=10, MIN_INLIERS=10)
                if bounding_box is not None:
                    match_bounding_boxes.append(bounding_box)
                    match_labels.append(label)
                    break
        return match_bounding_boxes, match_labels
