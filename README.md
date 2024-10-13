
A new model based on the UNet framework is proposed, which embeds Swin transformer blocks and channel attention mechanism (CAM) parallel modules for glacier extraction.

# 1 Environment 
To set up the environment, first install Anaconda. 
Then, open Anaconda Prompt and create a virtual environment by running conda create -n pytorch python=3.8, followed by activating it with conda activate pytorch.
 Install PyTorch by selecting the appropriate command for your CUDA version from PyTorch Previous Versions; for CUDA 11.3, use conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch. 
Next, install other essential libraries with the commands: conda install gdal, conda install matplotlib, conda install scipy, conda install tqdm, and the respective pip installs for einops, tensorboardX, timm, medpy, SimpleITK, yacs, and scikit-learn. 
Common commands include checking the environment with activate pytorch, switching back to the base environment with activate root, listing all environments with conda info --env, viewing installed libraries with conda list,
 and removing an environment with conda env remove --name your_environment_name or all packages in a virtual environment with conda remove -n your_env_name --all.

 # 2 Prepare data
We provide a small sample dataset， Contains sliced images and labels[Get processed data in this link]（[https://zenodo.org/records/13923320](https://zenodo.org/records/13923320)）.The band synthesis sequence of the image is: B2 of Sentinel-2, B3,B4,B8,B12,NDVI,NDWI,NDSI,DEM,Slope, Sentinel-2 VV band.

# 3 preprocessing
Preprocessing includes Python scripts for preprocessing remote sensing images, particularly croc_label.py for cropping labels, where 0 represents background, 1 represents glacier, and gdal_crop.py is used for cropping remote sensing images and unifying data dimensions. The sample images are cropped into 11 bands of 224 * 224 size, with data dimensions unified to 0-255.
![image](https://github.com/user-attachments/assets/5f034d5c-5f69-4c0c-9259-c3c663b2603c)# Trans-CAUNet
# 4 train/test/predict
Train. py, evaluate. py, and predict_totif. py are used to train models, test model accuracy, and predict remote sensing images. The parameters have been set by default, and you can directly run them by modifying the input and output paths of the data.

# 5 mosaic
The predicted image is 224 * 224 in size, and it needs to be stitched together to restore the original image size. Run gdal_combine.py to stitch the images together to obtain the complete predicted image.
![image](https://github.com/user-attachments/assets/b520da98-657c-47df-913a-5c702af296be)

