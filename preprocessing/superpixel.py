
import numpy as np
import cv2
import math
from torchvision import transforms
from skimage.transform import resize 
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries 

class SuperpixelProcessor:
    def __init__(self, slide, slide_name, mask_generator=None):
        """
        """

    def superpixel_segmenting(
        obj, downsample_size = 1096, n_segments=2000, compactness=10.0, start_label=0):
        # start = time.time()
        (
            downsample_factor, 
            new_width, 
            new_height, 
            curr_width, 
            curr_height
        ) = self._rescaling_stat_for_segmentation(
            obj, downsample_size)

        # Downscale the region and prepare for mask generation
        downscaled_region = self._downscaling(
            obj, new_width, new_height)
        downscaled_region_array = np.array(downscaled_region)

        lab_image = color.rgb2lab(downscaled_region_array)
        superpixel_labels = segmentation.slic(
            lab_image, 
            n_segments=n_segments, 
            compactness=compactness, 
            start_label=start_label
            )

        # print((time.time()-start)/60.00)

        return (
            superpixel_labels, 
            segmented_mask, 
            downsample_factor, 
            new_width, 
            new_height, 
            downscaled_region_array, 
            lab_image
            ) 

    @staticmethod 
    def _downscaling(obj, new_width, new_height):
        """
        Downscale the given object (image or slide) to the specified size.
        """
        if isinstance(obj, np.ndarray):  # If it's a NumPy array
            # Resize using scikit-image (resize scales and interpolates)
            image_numpy = resize(obj, (new_height, new_width), anti_aliasing=True)
            image_numpy = (image_numpy * 255).astype(np.uint8)

        elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
            obj = obj.resize((new_width, new_height))
            image_numpy = np.array(obj)

        elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
            thumbnail = obj.get_thumbnail((new_width, new_height))
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts the image to a tensor (C, H, W)
            ])
            image_tensor = transform(thumbnail)
            image_numpy = image_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C) numpy format
        else:
            raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")

        return image_numpy 

    @staticmethod 
    def _rescaling_stat_for_segmentation(
        obj, downsampling_size=1024):
        """
        Rescale the image to a new size and return the downsampling factor.
        """
    
        if hasattr(obj, 'shape'):
            original_width, original_height = obj.shape[:2]
        elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
            original_width, original_height = obj.size
        elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
            original_width, original_height = obj.dimensions
        else:
            raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")
        
        if original_width > original_height:
            downsample_factor = int(downsampling_size * 100000 / original_width) / 100000
        else:
            downsample_factor = int(downsampling_size * 100000 / original_height) / 100000
        
        new_width = int(original_width * downsample_factor)
        new_height = int(original_height * downsample_factor)
        
        return downsample_factor, new_width, new_height, original_width, original_height 

    def identify_foreground_background(
        equalized_image, superpixel_labels, threshold=240): 
        equalized_image = np.array(equalized_image)
        unique_superpixels = np.unique(superpixel_labels)
        background_mask = np.all(equalized_image >= threshold, axis=-1)  # RGB close to white

        foreground_superpixels = []
        background_superpixels = []

        for label in unique_superpixels:
            superpixel_mask = superpixel_labels == label
            superpixel_background = np.sum(
                background_mask[superpixel_mask]) / np.sum(superpixel_mask)

            if superpixel_background > 0.5:
                background_superpixels.append(label)
            else:
                foreground_superpixels.append(label)

        return foreground_superpixels, background_superpixels
    
    def get_bounding_boxes_for_foreground_segments(
        original_image, superpixel_labels, foreground_superpixels):
        if original_image.dtype != np.uint8:
            original_image = np.uint8(original_image)  # Convert to uint8

        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        output_image = np.copy(original_image)

        bounding_boxes = {}

        for label in foreground_superpixels:
            coords = np.column_stack(np.where(superpixel_labels == label))

            ymin, xmin = np.min(coords, axis=0)
            ymax, xmax = np.max(coords, axis=0)

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            bounding_boxes[label] = (xmin, ymin, xmax, ymax)

        return bounding_boxes, output_image 
    
    def get_bounding_boxes_for_foreground_segments(
        original_image, superpixel_labels, foreground_superpixels):
        if original_image.dtype != np.uint8:
            original_image = np.uint8(original_image)  # Convert to uint8

        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        output_image = np.copy(original_image)

        bounding_boxes = {}

        for label in foreground_superpixels:
            coords = np.column_stack(np.where(superpixel_labels == label))

            ymin, xmin = np.min(coords, axis=0)
            ymax, xmax = np.max(coords, axis=0)

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            bounding_boxes[label] = (xmin, ymin, xmax, ymax)

        return bounding_boxes, output_image

 