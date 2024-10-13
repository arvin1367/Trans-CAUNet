#Environment configuration
To set up the environment, first install Anaconda and Visual Studio Code. 
Then, open Anaconda Prompt and create a virtual environment by running conda create -n pytorch python=3.8, followed by activating it with conda activate pytorch.
 Install PyTorch by selecting the appropriate command for your CUDA version from PyTorch Previous Versions; for CUDA 11.3, use conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch. 
Next, install other essential libraries with the commands: conda install gdal, conda install matplotlib, conda install scipy, conda install tqdm, and the respective pip installs for einops, tensorboardX, timm, medpy, SimpleITK, yacs, and scikit-learn. 
The preprocessing folder includes Python scripts for preprocessing remote sensing images, specifically crop_label.py for cropping labels and gdal_crop.py for cropping images and standardizing dimensions. Evaluate.py is used to test the accuracy of the model, predict_totit.py is used for prediction, and train.by is used for training the model.
Common commands include checking the environment with activate pytorch, switching back to the base environment with activate root, listing all environments with conda info --env, viewing installed libraries with conda list,
 and removing an environment with conda env remove --name your_environment_name or all packages in a virtual environment with conda remove -n your_env_name --all.
