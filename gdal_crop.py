########################################################################################################################
# Image Cropping Script
########################################################################################################################

from cmath import nan
import time  # Import time module for timing
import numpy as np  # Import numpy library for array operations
from osgeo import gdal  # Import GDAL library for handling geospatial data
import os

class GRID:

    def load_image(self, filename):
        """
        Load image function

        Parameters:
            filename (str): File path of the input image

        Returns:
            img_proj (str): Projection information of the image
            img_geotrans (tuple): Geotransformation information of the image
            img_data (numpy.ndarray): Data array of the image
        """
        image = gdal.Open(filename)  # Open the image file using GDAL

        img_width = image.RasterXSize  # Get image width
        img_height = image.RasterYSize  # Get image height

        img_geotrans = image.GetGeoTransform()  # Get the geotransformation information
        img_proj = image.GetProjection()  # Get the projection information
        img_data = image.ReadAsArray(0, 0, img_width, img_height)  # Read the image data as an array

        del image  # Release the image data object
        return img_proj, img_geotrans, img_data  # Return projection information, geotransformation, and data array

    def write_image(self, filename, img_proj, img_geotrans, img_data):
        # Determine the dimensions of the image
        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands = 1
            img_height, img_width = img_data.shape

        # Get the TIFF format driver
        driver = gdal.GetDriverByName('GTiff')

        # Create a new image file
        image = driver.Create(filename, img_width, img_height, img_bands, gdal.GDT_Byte)

        # Set the geotransformation information of the image
        image.SetGeoTransform(img_geotrans)

        # Set the projection information of the image
        image.SetProjection(img_proj)

        # Write each band to the file
        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data[0])
        else:
            for i in range(img_bands):
                band_data = img_data[i]
                image.GetRasterBand(i + 1).WriteArray(band_data)

        # Release the image data object
        del image

    def normalize_band(self, band, nodata_value=-3.402823e+38):
        MI = np.min(band)
        # Create a mask to mark no-data values
        mask = (band < -1000000) | np.isnan(band)
        # Use the mask to handle no-data values
        band = np.ma.masked_array(band, mask)

        # Handle the minimum and maximum values
        band_min = band.min()
        band_max = band.max()

        if band_max - band_min == 0:
            # If all valid values are equal, return directly
            normalized_band = np.zeros_like(band, dtype=np.uint8)
        else:
            # Normalization process
            normalized_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)

        # Set the no-data value area back to the original value
        normalized_band[mask] = 0
        return normalized_band

if __name__ == '__main__':

    path_img = r"\image.tif"  # Input image path
    path_out = r"\Images/"  # Output image folder path
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    t_start = time.time()  # Record start time

    run = GRID()  # Create an instance of the GRID class
    proj, geotrans, data = run.load_image(path_img)  # Load input image data

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)

    channel, height, width = data.shape  # Get the number of channels, height, and width of the image data

    # Normalize each band first
    for c in range(channel):
        data[c] = run.normalize_band(data[c])

    patch_size_w = 224  # Define the width of the cropped patch
    patch_size_h = 224  # Define the height of the cropped patch

    num = 0  # Initialize the image numbering

    for i in range(height // patch_size_h + 1):  # Iterate over patches in the height direction
        for j in range(width // patch_size_w + 1):  # Iterate over patches in the width direction
            num += 1  # Update the image numbering

            sub_image = data[:, i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w]  # Extract the patch

            # Calculate the geotransformation for the cropped patch
            px = geotrans[0] + j * patch_size_w * geotrans[1] + i * patch_size_h * geotrans[2]
            py = geotrans[3] + j * patch_size_w * geotrans[4] + i * patch_size_h * geotrans[5]
            new_geotrans = [px, geotrans[1], geotrans[2], py, geotrans[4], geotrans[5]]

            # Write the cropped image data to a new file
            run.write_image(path_out + '{}.tif'.format(num), proj, new_geotrans, sub_image)

            # Output the image processing information
            time_end = time.time()
            print('Image {} processed, elapsed time: {} seconds'.format(num, round((time_end - t_start), 4)))

    t_end = time.time()  # Record end time
    print('All images processed, total time: {} seconds'.format(round((t_end - t_start), 4)))  # Output the total processing time
