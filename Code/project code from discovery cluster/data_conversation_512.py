## This file will resize the image for Melanoma classification to (512*512)


# imports
 
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image, ImageFile


# path

input_folder = "/scratch/dabas.a/data/jpeg/train/"
output_folder = "/scratch/dabas.a/data/jpeg/train_resize"

# the function to resize the image to any given size 

def resize_image(image_path, output_folder,resize):
    
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder,base_name)
    
    img = Image.open(image_path)
    img = img.resize( (resize[1], resize[0]), resample = Image.BILINEAR )
    
    img.save(outpath)

# lets see how many images we have

images = glob.glob(os.path.join(input_folder,"*.jpg"))

Parallel(n_jobs=20)(
    delayed(resize_image)(
    i, output_folder,(512,512)
    )for i in tqdm(images)
)


## Printing the results

# original JPG Images
len1 = []
for i in tqdm(images):
    len1.append(i)
print("Number of original Images",len(len1))

# Modified JPG Images
out_images = glob.glob(os.path.join(output_folder,"*.jpg"))
len2 = []
for i in tqdm(out_images):
    len1.append(i)
print("Number of resized Images",len(len2))
