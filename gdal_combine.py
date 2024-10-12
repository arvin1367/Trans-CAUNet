########################################################################################################################
# Image Merging Script
########################################################################################################################

import os  # Import the os module for file path operations
import glob  # Import the glob module to get a list of files
from osgeo import gdal  # Import the GDAL library for handling geographic data
from math import ceil  # Import the math module for ceiling calculations
from tqdm import tqdm  # Import tqdm for displaying a progress bar

def GetExtent(infile):
    """
    Function to get the geographic extent of raster data

    Parameters:
        infile (str): Input file path

    Returns:
        min_x, max_y, max_x, min_y (float): Geographic extent of the data
    """
    ds = gdal.Open(infile)  # Open the data file using GDAL
    geotrans = ds.GetGeoTransform()  # Get the geographic transformation info
    xsize = ds.RasterXSize  # Get the raster width
    ysize = ds.RasterYSize  # Get the raster height
    min_x, max_y = geotrans[0], geotrans[3]  # Calculate the top-left corner coordinates
    max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]  # Calculate the bottom-right corner coordinates
    ds = None  # Release the dataset
    return min_x, max_y, max_x, min_y  # Return the geographic extent

def RasterMosaic(file_list, outpath):
    """
    Function to merge raster data

    Parameters:
        file_list (list): List of input files
        outpath (str): Output file path
    """
    Open = gdal.Open

    min_x, max_y, max_x, min_y = GetExtent(file_list[0])  # Get the geographic extent of the first file
    for infile in file_list:
        minx, maxy, maxx, miny = GetExtent(infile)  # Get the extent of each file
        min_x, min_y = min(min_x, minx), min(min_y, miny)  # Update the top-left corner coordinates
        max_x, max_y = max(max_x, maxx), max(max_y, maxy)  # Update the bottom-right corner coordinates

    in_ds = Open(file_list[0])  # Open the first file
    in_band = in_ds.GetRasterBand(1)  # Get the first band of the first file

    geotrans = list(in_ds.GetGeoTransform())  # Get the geographic transformation info
    width, height = geotrans[1], geotrans[5]  # Get pixel width and height
    columns = ceil((max_x - min_x) / width)  # Calculate the number of columns for the output file
    rows = ceil((max_y - min_y) / (-height))  # Calculate the number of rows for the output file

    driver = gdal.GetDriverByName('GTiff')  # Get the TIFF driver
    # Create the output dataset
    out_ds = driver.Create(outpath, columns, rows, 1, in_band.DataType, options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    out_ds.SetProjection(in_ds.GetProjection())  # Set the projection info
    geotrans[0] = min_x  # Set the top-left X coordinate
    geotrans[3] = max_y  # Set the top-left Y coordinate
    out_ds.SetGeoTransform(geotrans)  # Set the geographic transformation info
    inv_geotrans = gdal.InvGeoTransform(geotrans)  # Get the inverse of the geographic transformation

    for in_fn in tqdm(file_list):  # Iterate through the list of files
        in_ds = Open(in_fn)  # Open the file
        in_gt = in_ds.GetGeoTransform()  # Get the geographic transformation info
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])  # Calculate the offset

        x, y = map(int, offset)  # Convert the offset to integers
        for i in range(1):  # Iterate through the bands (change this for multi-band images)
            data = in_ds.GetRasterBand(i+1).ReadAsArray()  # Read the band data
            out_ds.GetRasterBand(i+1).WriteArray(data, x, y)  # Write the data to the output dataset

    del in_ds, out_ds  # Release the datasets

if __name__ == '__main__':
    image_path = r"\image path"  # Path to the images to be merged
    result_path = r"\result save path"  # Path for the output file
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    imageList = glob.glob(image_path + "/*.tif")  # Get the list of image files
    result = os.path.join(result_path, "result.tif")  # Path for the merged output file
    RasterMosaic(imageList, result)  # Perform raster merging
