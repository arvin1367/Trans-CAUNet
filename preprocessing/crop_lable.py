import time  # Import time module for timing
import os  # Import os module for file path operations
import numpy as np  # Import numpy library for array operations
from osgeo import gdal  # Import GDAL library for handling geospatial data

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
        try:
            image = gdal.Open(filename)  # Open the image file using GDAL
            if image is None:
                raise FileNotFoundError(f"Unable to open the image file: {filename}")

            img_width = image.RasterXSize  # Get image width
            img_height = image.RasterYSize  # Get image height

            img_geotrans = image.GetGeoTransform()  # Get the geotransformation information
            img_proj = image.GetProjection()  # Get the projection information
            img_data = image.ReadAsArray(0, 0, img_width, img_height)  # Read the image data as an array

            # # Check for NODATA values, set NODATA values to 0
            # for band in range(image.RasterCount):
            #     band_data = image.GetRasterBand(band + 1)
            #     nodata_value = band_data.GetNoDataValue()
            #     if nodata_value is not None:
            #         img_data[img_data == nodata_value] = 0

            del image  # Release the image data object
            return img_proj, img_geotrans, img_data  # Return projection, geotransformation, and data array
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

    def write_image(self, filename, img_proj, img_geotrans, img_data):
        """
        Write image function

        Parameters:
            filename (str): File path of the output image
            img_proj (str): Projection information of the image
            img_geotrans (tuple): Geotransformation information of the image
            img_data (numpy.ndarray): Data array of the image
        """
        try:
            img_data[img_data == 3] = 0  # Set pixel values of 3 to 0
            if len(img_data.shape) == 3:
                img_bands, img_height, img_width = img_data.shape
            else:
                img_bands, (img_height, img_width) = 1, img_data.shape
            
            driver = gdal.GetDriverByName('GTiff')  # Get the TIFF format driver
            image = driver.Create(filename, img_width, img_height, img_bands, gdal.GDT_Byte)  # Create a new image file

            image.SetGeoTransform(img_geotrans)  # Set the geotransformation information
            image.SetProjection(img_proj)  # Set the projection information

            if img_bands == 1:
                image.GetRasterBand(1).WriteArray(img_data)  # Write single-band image data
            else:
                for i in range(img_bands):
                    image.GetRasterBand(i + 1).WriteArray(img_data[i])  # Write multi-band image data

            del image  # Release the image data object
        except Exception as e:
            print(f"Error writing image: {e}")
            raise

if __name__ == '__main__':
    path_img = r"\label.tif"  # Input image path
    path_out = r"output path\Masks/"  # Output image folder path
    if not os.path.exists(path_out):
        os.makedirs(path_out) 

    t_start = time.time()  # Record start time

    run = GRID()  # Create an instance of the GRID class
    proj, geotrans, data = run.load_image(path_img)  # Load the input image data

    height, width = data.shape  # Get the height and width of the image data

    patch_size_w = 224  # Define the width of the cropped patch
    patch_size_h = 224  # Define the height of the cropped patch

    num = 0  # Initialize the image numbering

    for i in range(height // patch_size_h + 1):  # Iterate over patches in the height direction
        for j in range(width // patch_size_w + 1):  # Iterate over patches in the width direction
            num += 1  # Update the image number

            sub_image = data[i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w]  # Extract the patch

            # Calculate the geotransformation for the cropped patch
            px = geotrans[0] + j * patch_size_w * geotrans[1] + i * patch_size_h * geotrans[2]
            py = geotrans[3] + j * patch_size_w * geotrans[4] + i * patch_size_h * geotrans[5]
            new_geotrans = [px, geotrans[1], geotrans[2], py, geotrans[4], geotrans[5]]

            # Write the cropped patch to a new file
            output_file = os.path.join(path_out, f'{num}.tif')
            run.write_image(output_file, proj, new_geotrans, sub_image)

            # Output the processing information
            time_end = time.time()
            print(f'Image {num} processed, elapsed time: {round((time_end - t_start), 4)} seconds')

    t_end = time.time()  # Record end time
    print(f'All images processed, total time: {round((t_end - t_start), 4)} seconds')  # Output the total processing time
