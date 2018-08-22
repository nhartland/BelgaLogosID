#!python3
import re
import os
import pandas as pd
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PIL import Image


# BelgaLogos webpage
webpage_url = "http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html"

# Annotation/metadata filename
data_folder = "data"
qset3_internal_and_local_gt_file = "qset3_internal_and_local.gt"

# Column names for metadata file
qset3_internal_and_local_gt_headers = [
    "brand", "image_file", "type", "ok",
    "bbx1", "bby1", "bbx2", "bby2"]

# Asserts types for columns, caught malformed rows in metadata file
qset3_internal_and_local_gt_dtypes = {
    "brand": str,
    "image_file": str,
    "type": str,
    "ok": bool,
    "bbx1": np.int64,
    "bby1": np.int64,
    "bbx2": np.int64,
    "bby2": np.int64}

# Categorisation of logo labels according to category
LOGO_TYPE = {
    "Adidas": "clothing",
    "Adidas-text": "clothing",
    "Airness": "clothing",
    "Base": "NA",
    "BFGoodrich": "NA",
    "Bik": "NA",
    "Bouigues": "NA",
    "Bridgestone": "NA",
    "Bridgestone-text": "NA",
    "Carglass": "NA",
    "Citroen": "car",
    "Citroen-text": "car",
    "CocaCola": "NA",
    "Cofidis": "NA",
    "Dexia": "NA",
    "ELeclerc": "NA",
    "Ferrari": "car",
    "Gucci": "clothing",
    "Kia": "car",
    "Mercedes": "car",
    "Nike": "clothing",
    "Peugeot": "car",
    "Puma": "clothing",
    "Puma-text": "clothing",
    "Quick": "NA",
    "Reebok": "clothing",
    "Roche": "NA",
    "Shell": "NA",
    "SNCF": "NA",
    "Standard_Liege": "NA",
    "StellaArtois": "NA",
    "TNT": "NA",
    "Total": "NA",
    "Umbro": "clothing",
    "US_President": "NA",
    "Veolia": "NA",
    "VRT": "NA",
}


def read_metadata():
    """ Reads the BelgaLogos metadata/annotations file for a specific feature category """
    filename = os.path.join(data_folder, qset3_internal_and_local_gt_file)
    metadata = pd.read_csv(filename, sep='\t',
                           lineterminator='\n',
                           header=None,
                           names=qset3_internal_and_local_gt_headers,
                           dtype=qset3_internal_and_local_gt_dtypes)
    # Check for NULL entries in all columns
    nulltest = metadata.isnull().any()
    for field_is_null in nulltest:
        if field_is_null:
            assert "NULL entry in read_metadata"
    # Further annotate with brand category (e.g car, clothing)
    category = metadata.apply(lambda row: LOGO_TYPE[row['brand']], axis=1)
    metadata['category'] = category

    return metadata


def scrape_testdata():
    """ This function scrapes the BelgaLogos webpage for their 'canonical' image
    counts, the data is returned as a DataFrame and can be used for testing"""

    # BeautifulSoup-ify the webpage
    html = urlopen(webpage_url)
    soup = BeautifulSoup(html, features="lxml")

    # Scrape out the relevant tables (there are two) and append to a list of dataframes
    frames = []
    for tag in soup.find_all(text=re.compile('#Junk')):
        parent_table = tag.findParent('table')
        # Might be able to do this neater by applying read_html to all tables at once
        df = pd.read_html(str(parent_table), header=0, index_col=False)[0]
        df = df.drop(['Illustration'], axis=1)  # Don't need the illustration column
        df = df.set_index('Logo name')
        df = df.infer_objects()  # Convert to numeric types
        frames.append(df)

    total_df = pd.concat(frames)
    return total_df


def filter_by_boundingbox(metadata, min_dim, max_dim):
    """"
    This filters an dataset (a pandas DataFrame from read_metadata), ensuing it
    only contains bounding-boxes with:

            min_dim < (height or width) <= max_dim

    """
    # Shortcut
    ds = metadata
    # Ensure all bounding-boxes are at least {min_dim} pixels wide and high
    condition = (ds.bbx2-ds.bbx1 >  min_dim) & (ds.bby2 - ds.bby1 > min_dim) \
              & (ds.bbx2-ds.bbx1 <= max_dim) & (ds.bby2 - ds.bby1 <= max_dim)
    filtered_ds = ds[condition]
    return filtered_ds


def load_bb_images(metadata):
    """
        Given a pandas DataFrame containing the BelgaLogos metadata, return a
        new DataFrame consisting of the corresponding bounded-box images
        labelling the logo.
    """

    def load_image(row):
        filename = os.path.join(data_folder, 'images', row['image_file'])
        im = Image.open(filename).convert('RGB')
        bb = im.crop((row['bbx1'], row['bby1'], row['bbx2'], row['bby2']))
        return bb

    images = metadata.apply(load_image, axis=1)
    return images
