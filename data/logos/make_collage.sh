#!/bin/bash
mkdir conv
for image in *.jpg; do
    convert -geometry 300x $image conv/$image
done

montage conv/*.jpg -geometry +2+2 collage.jpeg
rm -rf conv
