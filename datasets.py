import os
import random
#import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from osgeo import gdal
from PIL import Image
from torchvision import transforms as T
import glob 

class ToTensor(object):
    def __call__(self, image, target):
        # Convert the image to a tensor
        image = torch.from_numpy(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)  # Randomly choose rotation angle
    image = np.rot90(image, k)    # Rotate the image
    label = np.rot90(label, k)    # Rotate the label
    axis = np.random.randint(0, 2)  # Randomly choose flip axis
    image = np.flip(image, axis=axis).copy()  # Flip the image
    label = np.flip(label, axis=axis).copy()  # Flip the label
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)  # Randomly choose rotation angle
    image = ndimage.rotate(image, angle, order=0, reshape=False)  # Rotate the image
    label = ndimage.rotate(label, angle, order=0, reshape=False)  # Rotate the label
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:  # With a 50% chance, perform random rotation and flipping
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:  # With a 50% chance, perform random rotation
            image, label = random_rotate(image, label)
        x, y = image.shape
        # Check if the image size matches the output size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # Resize the image
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # Resize the label
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # Convert to tensor and add dimension
        label = torch.from_numpy(label.astype(np.float32))  # Convert label to tensor
        sample = {'image': image, 'label': label.long()}  # Create sample dictionary
        return sample

class dataset(Dataset):
    def __init__(self, root, type, transform=None):
        self.type = type
        if self.type == 'train' or self.type == 'test':
            assert os.path.exists(root), "Path '{}' does not exist.".format(root)
            image_dir = os.path.join(root, 'Images')
            mask_dir = os.path.join(root, 'Masks')
            # txt_path = os.path.join(root, txt_name)
            # assert os.path.exists(txt_path), "File '{}' does not exist.".format(txt_path)

            # with open(os.path.join(txt_path), "r") as f:
            #     file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

            self.images = glob.glob(os.path.join(image_dir, "**", "*.tif"), recursive=True)  #[os.path.join(image_dir, x + ".tif") for x in file_names]
            self.masks = glob.glob(os.path.join(mask_dir, "**", "*.tif"), recursive=True)  #[os.path.join(mask_dir, x + ".tif") for x in file_names]
            assert (len(self.images) == len(self.masks))  # Ensure the number of images matches the number of masks
            self.transforms = transform
            self.totensor = ToTensor()
        else:
            assert os.path.exists(root), "Path '{}' does not exist.".format(root)
            image_dir = os.path.join(root, 'Images')
            self.images = glob.glob(os.path.join(image_dir, "**", "*.tif"), recursive=True)
            self.transforms = transform
            self.totensor = ToTensor()

        # self.resize = T.Resize(224)

    def __len__(self):
        return len(self.images)  # Return the size of the dataset
    
    def __getitem__(self, idx):
        if self.type == 'train' or self.type == 'test':
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is the image segmentation.
            """
            dataset = gdal.Open(self.images[idx])
            # Get the number of columns in the matrix
            self.width = dataset.RasterXSize
            # Get the number of rows in the raster matrix
            self.height = dataset.RasterYSize
            # Get the data
            img = dataset.ReadAsArray(0, 0, self.width, self.height)  # Read image data
            shape = img.shape
            target = Image.open(self.masks[idx])  # Open the corresponding mask

            if self.transforms is not None:
                img, target = self.transforms(img, target)  # Apply transformations
            img, target = self.totensor(img, target)  # Convert to tensors
            a = img.shape
            target = target.unsqueeze(0)  # Add dimension to the target
            # target = self.resize(target).float()
            b = target.shape
            return img, target
        else:
            dataset = gdal.Open(self.images[idx])
            # Get the number of columns in the matrix
            self.width = dataset.RasterXSize
            # Get the number of rows in the raster matrix
            self.height = dataset.RasterYSize
            img_geotrans = dataset.GetGeoTransform()  # Get the geographical transformation information of the image
            img_geotrans = np.array(img_geotrans)
            img_proj = dataset.GetProjection()  # Get the projection information of the image
            # Get the data
            img = dataset.ReadAsArray(0, 0, self.width, self.height)  # Read image data
            image = torch.from_numpy(img)  # Convert to tensor
            return image, os.path.basename(self.images[idx]), img_geotrans, img_proj  # Return image and geographical information
