#!/bin/bash
# Get images
wget http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/images.tar.gz && tar -xvzf images.tar.gz
# Cleanup
rm images.tar.gz
