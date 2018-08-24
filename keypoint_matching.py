# Functions for performing keypoint matching with openCV
# The brute-force matching code here was (heavily) modified from the original source:
# https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects

import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

__DEBUG__ = False  # Debug flag for verbose output


def find_and_plot_keypoints(image, finder):
    """
        Takes an OpenCV image and an OpenCV feature finder, computes the
        keypoints ands returns an annotated image showing the keypoint
        locations, along with a count of the computed keypoints.
    """
    keypoints, descriptors = finder.detectAndCompute(image, None)
    keypoint_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return keypoint_image, len(keypoints)


def meanshift_keypoint_labels(keypoints, quantile):
    """
        Determines the clustering (with meanshift) for a set of keypoints
        and a specified bandwidth quantile. Returns a list of labels for the keypoints,
        specifying which cluster they lie in, and the total number of clusters.
    """
    # Add points to list
    point_locations = np.array([ kp.pt for kp in keypoints])

    # Compute mean-shift clusters
    bandwidth = estimate_bandwidth(point_locations, quantile=quantile)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(point_locations)
    return ms.labels_,  max(ms.labels_)+1


def meanshift_keypoint_clusters(image, finder, quantile=0.02):
    """
        Takes an OpenCV image and a keypoint finder. Generates the keypoints
        for the image, and clusters them through the MeanShift clustering
        algorith. Returns a list of keypoint clusters and a corresponding list
        of descriptor clusters. Optionally the quantile for the MS bandwidth
        estimation can be specified.
    """
    # Compute keypoints and descriptors
    keypoints, descriptors = finder.detectAndCompute(image, None)
    # Get cluster labels through meanshift clustering
    labels, nclusters = meanshift_keypoint_labels(keypoints, quantile=quantile)

    # Return list of clusters, containing the keypoints
    kp_clusters = [None] * nclusters  # Keypoint clusters
    ds_clusters = [None] * nclusters  # Descriptor clusters
    keypoints = np.array(keypoints)  # To be able to use the output of 'where' as an index
    for i in range(nclusters):
        d, = np.where(labels == i)  # Point indices which are in cluster 'i'
        kp_clusters[i] = keypoints[d]
        ds_clusters[i] = descriptors[d]

    return kp_clusters, ds_clusters


def build_bounding_box(img1, img2, src_pts, dst_pts, MIN_INLIERS):
    """
        Given a template image (img1) and a test image (img2) along with
        a set of matched keypoints from img1 (src_pts) and img2 (dst_pts), this
        function computes a homography transformation between the two pointsets.
        This is then used to transform the bounding-box of the template image
        into the space of the test image.

        This bounding-box is returned in a form suitable for drawing with cv2.polyLines.

        When specified, MIN_INLIERS requires that the homography finder has at least
        MIN_INLIERS point matches that can be used to construct a 'good fit' homography.
    """

    # Find the homography between source and destination points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

    if MIN_INLIERS is not None:
        # Get number of inliers: matches that fit the homography
        inlier_count = mask.ravel().sum()
        # Not enough high-quality points for homography transformation
        if inlier_count < MIN_INLIERS:
            return None

    if M is None:
        # No transformation found
        return None
    else:
        h, w, _ = img1.shape
        pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        return [np.int32(dst)]


def get_matching_boundingbox(template_img, test_img,
                    template_kp, test_kp, template_desc, test_desc,
                    matcher, MIN_MATCHES, MIN_INLIERS):
    """
        This function performs a matching between two sets of keypoint descriptors.
        If successful, it returns the bounding-box of the matched keypoints in the target image.
        Takes as arguments
            template_img:  The OpenCV image of the template to be matched.
            test_img:      The OpenCV image of the test for matching to the template
            template_kp:   The keypoints of the template image
            test_kp:       The keypoints of the test image
            template_desc: The keypoint descriptors of the template image
            test_desc:     The keypoint descriptors of the test image
            matcher:       The keypoint descriptors of the test image
    """
    # Perform match
    matches = matcher.match(template_desc, test_desc)
    if __DEBUG__ is True:
        print(len(matches) / min(len(test_kp), len(template_kp)), " matches found")

    # If there are sufficient matches, attempt to build a homography
    if len(matches) >= MIN_MATCHES:  # Minimum number of matches required
        # Get source and destination points for homography search
        src_pts = np.float32([ template_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([ test_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        bounding_box = build_bounding_box(template_img, test_img, src_pts, dst_pts, MIN_INLIERS)
        if bounding_box is not None:
            return bounding_box


def bruteforce_match_clusters(img1, img2, finder, norm,
                              QUANTILE=0.02, MIN_MATCHES=10, MIN_INLIERS=None):
    """
        This function attempts to locate all instances of the template image 'img1' in
        a test image 'img2', under affine transformation. This happens through several steps.

        1. Given a keypoint finder, the keypoints of the template are computed.
        2. The keypoints of the test image are computed and clustered with the MeanShift algorithm.
        3. For each cluster, it's keypoints are matched to their best-fit counterparts in the template keypoint set.
        4. The matches define a potential *homography* between the template and the cluster,
           this is used to transform the template dimensions into a proposed bounding-box in the test image space.
        5. The test image is annotated with the proposed bounding-boxes and returns.

        Optional arguments:
            QUANTILE: Sets the quantile used in mean-shift bandwidth approximator.
            MIN_MATCHES: Minimum number of succesful matches required between cluster and template.
            MIN_INLIERS: Minimum number of matches that fit the constructed homography.
    """

    # Compute keypoints and descriptors for template image
    template_keypoints, template_descriptors = finder.detectAndCompute(img1, None)
    # Compute keypoint and descriptor clusters for target image
    kp_clusters, ds_clusters = meanshift_keypoint_clusters(img2, finder, quantile=QUANTILE)

    # Set up brute-force matcher
    bf = cv2.BFMatcher(norm, crossCheck=True)

    match_bounding_boxes = []
    for ic in range(len(kp_clusters)):
        cluster_keypoints   = kp_clusters[ic]
        cluster_descriptors = ds_clusters[ic]
        bounding_box = get_matching_boundingbox(img1, img2,
                                                template_keypoints, cluster_keypoints,
                                                template_descriptors, cluster_descriptors,
                                                bf, MIN_MATCHES, MIN_INLIERS)
        if bounding_box is not None:
            match_bounding_boxes.append(bounding_box)

    img3 = img2.copy()
    for box in match_bounding_boxes:
        cv2.polylines(img3, box, True, 255, 3, cv2.LINE_AA)

    return img3
