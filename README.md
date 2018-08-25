# Brand recognition with the BelgaLogos dataset

The
[BelgaLogos](http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html)
dataset consists of 10,000 images from the [Belga](http://www.belga.be/) press
agency, annotated with the location and name of brand logos appearing in the
images. In total annotations for 37 separate logos are provided in the dataset,
covering a variety of businesses.

In this repository I study how this dataset can be applied to the training of a
logo-classification model, specifically targeting the subset of BelgaLogo
entries from automotive and fashion brands.

The analysis is mostly documented in [Jupyter](http://jupyter.org/) notebooks,
which document my analysis procedure step-by-step in detail. This repository is
not meant only as a store of results, but also as a record of the
thought-process followed in the analysis. 

## Contents

### Exploration of the dataset
1. [Parsing and basic data validation](parsing.ipynb)
2. [Initial study of the logo bounding-boxes](image_bb.ipynb)
3. [The automotive and fashion brand dataset](dataset.ipynb)

### Model survey and feasibility studies
1. [Survey of model options](models.ipynb)
2. [Keypoint matching feasibility study](keypoint.ipynb)

## Running the notebooks

The notebooks require the following python packages:
 - numpy
 - matplotlib
 - scikit-learn
 - beautifulsoup4 + lxml
 - OpenCV2
 - pillow

To fetch the source data for the analysis, run `data/getdata.sh` which will
download and decompress the BelgaLogos dataset images. 

**Note: this is a moderately large dataset (~1GB)**.
