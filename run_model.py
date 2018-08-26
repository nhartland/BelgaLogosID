#!/usr/bin/env python3
import lib.model as kpm
import argparse
import json, os, cv2, sys


def train_model_on_logos(model, logo_selection):
    """
        Trains a given model on all available logo templates in the directory
        data/{logo_selection}/ using a json file in that directory to specify
        what is available
    """
    logo_registration_path = os.path.join("data", logo_selection, "registered_logos.json")
    with open(logo_registration_path) as json_file:
        data = json.load(json_file)
        for logo in data:
            # Get the logo image
            logo_filename = os.path.join("data", logo_selection, logo.lower() + '.jpg')
            logo_image = cv2.imread(logo_filename)
            # Train the matcher on the image
            model.add_template(logo, logo_image)


def annotate_image(model, test_image):
    """
        Given a test_image and a model, find all object matches in the
        test_image and return an annotated version of it where test matches are
        displayed.
    """
    # Run the model over the image and validate the results with the above algorithm
    detected_objects = model.detect_objects(test_image)
    # Draw the images with a green box where there is a match, or a red box otherwise
    annotated_image = kpm.annotate_image_with_objects(test_image, detected_objects,
                                                      text_colour=(0, 155, 0))
    return annotated_image


def main(source, test_images):
    # Initialise model
    SIFT = cv2.xfeatures2d.SIFT_create()
    SIFTMatcher = kpm.KeypointMatcher(SIFT, cv2.NORM_L2SQR)
    train_model_on_logos(SIFTMatcher, source)

    for filename in test_images:
        test_image = cv2.imread(filename)
        result_image = annotate_image(SIFTMatcher, test_image)
        result_filename = "annotated_" + os.path.basename(filename)
        cv2.imwrite(result_filename, result_image)
        print("Exported ", result_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('template_source', help="choice of source templates", choices=['logos', 'live_logos'])
    parser.add_argument('images', help="list of input images", nargs='*')
    args = parser.parse_args()
    main(args.template_source, args.images)
