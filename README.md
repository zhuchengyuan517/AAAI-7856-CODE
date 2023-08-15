# AAAI-7856-CODE
AAAI-upload code
This project is the code for the signal preprocessing and recognition model mentioned in the paper.

It is divided into two folders: one is the preprocess folder, which includes four python files.

● split.py: A tool for slicing the raw data (time series signals) of distributed optical fiber sensing, dividing time series data into the required length.

● tf_pi.py: Used for batch converting timing signals into time-frequency maps.

● image_joint.py: Information fusion of time-frequency maps of continuous defense zones to generate space-time-frequency maps.

● data enhancement.py: Perform data augmentation on the input feature set of the model, including image flip, image scale and image blur.

The other is the model folder, which includes one json file, five python files and the pre-training weights file.

● model.py: Our threat estimation and recognition network model and its submodules.

● train.py: Main function of our threat estimation and recognition network framework. We can train the recognition network by python train.py based on default parameters.

● predict.py: We can can load the model test recognition network based on default parameters.

● my_dataset.py: Use for building custom datasets.

● utils.py: Other functions for model building. Besides, there are some functions for data reading, model evaluation and model train.

● class_indices.json: Define classification labels for recognition model.

● pre_efficientnetv2-s.pth: The pre-training weights for loading our model.
