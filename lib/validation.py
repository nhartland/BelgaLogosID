# validation.py
# code for the validation of the logo-detection model
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os


def metadata_to_AABB(md):
    """ Formats the axis-aligned bounding-box of a BelgaLogos metadata entry,
        returns [x1, y1, x2, y2] """
    return [md.bbx1, md.bby1, md.bbx2, md.bby2]


def vertices_to_AABB(points):
    """ Computes the axis-aligned bounding-box of a list of 2D points,
        returns [x1, y1, x2, y2] """

    minx, miny = np.min(points, axis=0)
    maxx, maxy = np.max(points, axis=0)
    return [minx, miny, maxx, maxy]


def compute_rectangle_area(r):
    """ Computes the area of a rectangle,
        supplied as [x1, y1, x2, y2]"""
    return (r[2] - r[0])*(r[3] - r[1])


def compute_rectangle_intersection(r1, r2):
    """ Computes the intersection area of two rectangles,
        supplied as [x1, y1, x2, y2]"""
    olx = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
    oly = max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    return olx * oly


def validate_detected_objects(metadata, detected_objects):
    """
        This function takes a pandas dataframe consisting of BelgaLogos
        metadata for a specific image, and attempts to match logo annotations
        to bounding-boxes detected by a model.  The `detected_objects` should
        be provided as a list of DetectedObject tuples (from kpm_model.py).

        Matching is checked by demanding at at least 20% of the
        axis-aligned-bounding -box (AABB) of the detected object overlaps with
        a corresponding annotation in the metadata.

        Returns a boolean mask for each element in `detected_objects`, True if
        the box is matched and false otherwise.

        # NOTE: At the moment this permits one annotation to match to multiple detected objects.
        # Notes for improvement: Instead of computing overlap between AABBs, compute actual
        polygon overlap with e.g pyclipper. This will naturally be slower, but also more accurate.
    """
    # List of bools, specifying if each detected object matches a corresponding annotation
    successful_match = [False] * len(detected_objects)
    for io, iobject in enumerate(detected_objects):
        # For now use the axis-aligned bounding box for overlap detection
        # The bounding-boxes in DetectedObject are in the format of vertices.
        detected = vertices_to_AABB(iobject.bounding_box)
        detected_area = compute_rectangle_area(detected)
        # Find matching-brand labels
        true_labels = metadata[metadata.brand == iobject.label]
        for index, true_annotation in true_labels.iterrows():
            itruth  = metadata_to_AABB(true_annotation)
            overlap = compute_rectangle_intersection(detected, itruth)
            # At least 20% of the detected area must overlap with
            # the true bounding-box to be matched
            if overlap > 0.2*detected_area:
                successful_match[io] = True
                break
    return successful_match


def study_matches(metadata, model):
    """
        For a provided list of image metadata and a matching model, count the
        number of true and false positives detected in the images.  Returns a
        pandas Series.
    """
    # Initialise counts
    true_positives   = 0
    actual_positives = 0
    false_positives  = 0
    image_count      = 0

    unique_images = metadata["image_file"].unique()
    for image in unique_images:
        # Fetch corresponding metadata for this image
        image_metadata = metadata[metadata["image_file"] == image]
        # Read the image file
        test_image = cv2.imread(os.path.join("data", "images", image))

        # Run the model over the image and validate the results with the above algorithm
        detected_objects = model.detect_objects(test_image)
        correct_matches = validate_detected_objects(image_metadata, detected_objects)

        actual_positives += len(image_metadata.index)
        true_positives   += np.sum(correct_matches)
        false_positives  += len(correct_matches) - np.sum(correct_matches)
        image_count += 1

    count_dict = { "true_positives": true_positives,
                   "actual_positives": actual_positives,
                   "true_positive_ratio": true_positives / actual_positives,
                   "false_positives": false_positives,
                   "false_positives_per_image": false_positives / image_count,
                   "image_count": image_count}

    return pd.Series(count_dict)


def validation_histogram(ax, results, labels,  plot_label="Model performance summary"):
    """
        This function generates a performance analysis histogram from a list
        of results from the `study_matches` function.

        Takes as input:
            ax: The matplotlib axis to draw to,
            results: The list of `study_matches` results,
            labels: A label for each of the entries in `results`
            (optional) plot_label: Titles the resulting plot.
    """
    # Set up plot data
    actual_positives = []
    true_positives = []
    false_positives = []

    for iresult, result in enumerate(results):
        ic = result.image_count
        actual_positives.append(result.actual_positives/ic)
        true_positives.append(result.true_positives/ic)
        false_positives.append(result.false_positives/ic)

    ind = np.arange(len(labels))
    barwidth = 0.35
    p1 = ax.bar(ind-barwidth, actual_positives, barwidth, color='b', bottom=0)
    p2 = ax.bar(ind, true_positives, barwidth, color='g', bottom=0)
    p3 = ax.bar(ind+barwidth, false_positives, barwidth, color='r', bottom=0)

    ax.set_xticks(ind + barwidth / 2)
    ax.set_xticklabels(labels)
    ax.legend((p1[0], p2[0], p3[0]), ('Real logos', 'True positives', 'False positives'))
    ax.yaxis.grid()
    ax.autoscale_view()
    ax.set_title(plot_label)
    ax.set_ylabel("Average count per image")
