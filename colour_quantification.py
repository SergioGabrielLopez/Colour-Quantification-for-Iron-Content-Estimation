# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:31:38 2022

@author: lopez
"""

"""This script analyses a folder containing images of seeds that have been stained with a dye that turns blue in the presence of iron. The
goal is to quantify the iron content based on the intensity of the blue color. That is done by transforming the image into an HSV image, where
HSV stands for Hue, Saturation, Value. the Hue channel is then used to select all the blue areas in the image. Then, the values in the Saturation
channel are analysed to quantify the intensity of the blue color.

The input must be a folder containing .tif, .jpg, or .jpeg images. 

The output is a series of images and a spreadsheet.

the images are:
    
(1) A relative iron content map that basically shows the Saturation values within the areas that are blue. The image also shows an outline of the 
seed in white.

(2) An image showing the area covered by the seed (in red) and the area containing iron (in blue).

(3) A black and white image that is colored only within the areas that show iron content.

The spreadsheet contains the following values:
    
(A) Image name.
(B) Percentage of the seed's area that is blue.
(C) Total area of the seed (in pixels squared).
(D) Area of the blue region (in pixels squared).
(E) Total saturation of the blue region.
(F) Total saturation of the blue region normalised by seed area."""


# Imports useful libraries.
from skimage import io, img_as_float, color, filters, morphology
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import glob

def get_filename(file):
    """This gets the name of the file."""
    string1 = os.path.splitext(file)[0]
    filename = os.path.basename(string1) # This gets the actual filename. 
    return filename


# Creates the dataframe to which the other dataframes will be appended.
great_dataframe = pd.DataFrame(columns=['Image name', "Percentage of the seed's area that is blue",'Total area of the seed (in pixels squared)','Area of the blue region (in pixels squared)','Total saturation of the blue region','Total saturation of the blue region normalised by seed area x 1000'])

# Creates a dialog window to obtain the folder where the images are.
root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory(title='Select the folder that contains the images.') 

# Opens each one of the images.
for ext in ('*.tif','*.jpg','*.jpeg'):
    for file in glob.glob(os.path.join(folder_selected, ext)):
        
        # Gets the actual name of the file.
        filename = get_filename(file)    
        
        # Reads the image.
        img = io.imread(file)
        
        # Transforms the image into a 64-bit image. 
        img = img_as_float(img) 
        
        # Transforms the image into an HSV (Hue Saturation Value) image.
        img_hsv = color.rgb2hsv(img)
        
        # Splits the HSV image.
        img_h = img_hsv[:,:,0] # Hue channel. This channel determines the color (e.g., red or blue or green)
        img_s = img_hsv[:,:,1] # Saturation channel. This channel determines the intensity of the color (e.g., intense blue or washed-out blue)
        img_v = img_hsv[:,:,2] # Value channel. This channel determines the  This channel determines the brightness of the pixels (e.g., very dark or very bright)
        
        # Gets the area of the entire seed (in pixels).
        thresh = filters.threshold_mean(color.rgb2gray(img))
        seed = color.rgb2gray(img) > thresh
        seed = morphology.remove_small_holes(seed,area_threshold=500)
        seed = morphology.remove_small_objects(seed,min_size=5000)
        area_seed = np.sum(seed)
        
        # Creates a mask that has True values outside the seed.
        background_seed = np.invert(seed)
        
        # Creates and empty image of the size of img.
        mask = np.zeros_like(img)
        
        # Creates a mask for the colour blue.
        mask[(img_h > 0.11) & (img_h < 0.80) & (img_v > 0.2) & (img_s > 0.14)] = 1
        
        # Deletes all the pixels in the mask that are outside the seed.
        mask[background_seed] = 0
        
        # Deletes small holes in the "blue" areas.
        mask = morphology.remove_small_holes(mask.astype(bool),5000) 
        
        # Deletes all the "blue" areas that are below a certain area in size.
        mask = morphology.remove_small_objects(mask.astype(bool),800)
        
        # Gets the area of the selected region (in pixels).
        area_selected = np.sum(mask[:,:,0])
        
        # Uses the mask to filter out all colors except the blue.
        img_selected = img*mask
        
        # Creates a grey image and adds color only within the mask.
        grey_img = color.gray2rgb(color.rgb2gray(np.copy(img)))
        grey_img[mask == 1] = img[mask == 1]
        
        # Creates a map of relative iron concentration.
        iron_map = img_s*mask[:,:,0]
        
        # Create a skeleton outline of the image.
        skeleton = filters.sobel(color.rgb2gray(img)) # uses a Sobel edge-detection algorithm to find the edges of the seed. 
        skeleton_mask = np.invert(morphology.remove_small_holes(np.array(mask[:,:,0],dtype=bool))) # Removes little empty specks from the areas in which there is iron and creates a mask from the result.
        skeleton = skeleton*skeleton_mask # This creates the skeleton outline with empty space in the areas in which there is iron. 
        
        # Uses the iron concentration map and the skeleton outline and plots them together. 
        plt.figure(figsize=(18,8))
        plt.imshow(iron_map,cmap='inferno')
        plt.axis('off')
        cbar = plt.colorbar()
        cbar.set_label('Relative iron content', rotation=270, fontsize=25, labelpad=50)
        cbar.ax.tick_params(labelsize=16)
        plt.contour(skeleton, cmap='Greys',alpha=0.5,linewidths=0.3)
        plt.tight_layout()
        plt.savefig(filename+'_iron_map.png',dpi=300)
        plt.close()
        
        # Displays the area of the entire seed with the selected areas superimposed.
        seed_RGB = np.zeros_like(img)
        area_RGB = np.zeros_like(img)
        seed_RGB[:,:,0] = seed
        area_RGB[:,:,2] = mask[:,:,0]
        seed_RGB[mask == 1] = area_RGB[mask == 1]
        plt.figure(figsize=(18,8))
        plt.imshow(seed_RGB)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename+'_selected_area.png',dpi=300)
        plt.close()
        
        # Displays the greyscale image with the selected area in color. 
        plt.figure(figsize=(18,8))
        plt.imshow(grey_img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename+'_greyscale.png',dpi=300)
        plt.close()
        
        # Percentage of the seed area that has iron in it. 
        area_percent_iron = (area_selected*100) / area_seed
        
        # Total sum of the saturation value in the area selected.
        total_saturation = np.sum(iron_map)
        
        # Sum of the saturation value in the area selected normalised by the area of the seed.
        normalised_saturation = total_saturation / area_seed
        
        # It adds the data to the dataframe.
        new_row = {'Image name':filename,"Percentage of the seed's area that is blue":area_percent_iron,'Total area of the seed (in pixels squared)':area_seed,'Area of the blue region (in pixels squared)':area_selected,'Total saturation of the blue region':total_saturation,'Total saturation of the blue region normalised by seed area x 1000':normalised_saturation*1000}
        
        great_dataframe = great_dataframe.append(new_row, ignore_index = True)
        
        
# Saves the great dataframe as an excel spreadsheet.
great_dataframe.to_excel('Data.xlsx')
        

    
    
    
    
    
    



