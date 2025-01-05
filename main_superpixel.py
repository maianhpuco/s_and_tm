import numpy as np
import cv2
import openslide 
from openslide import open_slide, ImageSlide 
from glob import glob
from skimage.segmentation import slic
from PIL import Image
from preprocessing import SuperpixelProcessor  # Assuming the class is in a file named your_module.py

SLIDE_PATH = './examples/wsi/'

# get the dataset for WSI images 

def main():
 
    processor = SuperpixelProcessor(slide_image, slide_name)

    # This would typically be a region you want to segment from the slide/image.
     

    # Define downsample size (e.g., 1024 pixels)
    downsample_size = 1024

    # Example parent bounding box (this is typically used in hierarchical segmentation)
    parent_absolute_bbox = [0, 0, slide_image.width, slide_image.height]

    # Call the segmentation method
    segmentation_output, downsample_factor, new_width, new_height, downscaled_region_array = processor.segmenting(
        obj, downsample_size, parent_absolute_bbox)

    # Display segmentation results
    print(f"Downsample Factor: {downsample_factor}")
    print(f"New Width: {new_width}, New Height: {new_height}")
    print("Segmentation Results:")
    for segment in segmentation_output:
        print(f"Segment {segment['idx']} - Bounding Box: {segment['abs_bbox']} - Area: {segment['square_area']}")

if __name__ == "__main__":
    print(SLIDE_PATH)
    wsi_format = 'tif'
    wsi_paths = glob.glob(f"./examples/wsi/*/*.{wsi_format}")
    print(wsi_paths)
