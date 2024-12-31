# CFPLncLoc
A novel deep learning model uses Chaos Game Representation (CGR) images of lncRNA sequences
and Multi-scale feature fusion (MFF) model Centralized Feature Pyramid (CFP) to predict
multi-label lncRNA subcellular localization.

## Requirements
python==3.8\
torch==1.10.0 or torch==1.13.1+cu117\
torchvision==0.11.1 or torchvision==0.14.1+cu117\
tensorflow==2.2.0 or tensorflow-gpu==2.2.0\
tensorboard==2.2.2\
keras==1.1.2\
pandas==1.1.0\
scipy==1.4.1\
protobuf==3.20.0\
opencv-python==4.10.0.84\
scikit-learn==1.3.2\
scikit-multilearn==0.2.0\
scikit-image==0.18.1\
numpy==1.19.0

## 1 Setup instructions
Install the dependencies corresponding to the Requirements;\
Download pre-trained model (resnet101) from https://download.pytorch.org/models/resnet101-5d3b4d8f.pth.

## 2 Parameter settings
### 2.1 Print scatter plot image
>pbaspect([1 1 1])                              % Sets the frame aspect ratio to 1:1:1\
>set(gca, 'LooseInset', get(gca, 'TightInset')) % Remove white spaces\
>set(gca, 'looseInset', [0 0 0 0])              % Completely remove the interval\
>set(gcf, 'color', 'w')                         % Sets the background color to white
### 2.2 Get three-channel data from image
>IMAGE_SIZE = 256  # pixels\
>GRAY_SCALE = False\
>CHANNELS = 1 if GRAY_SCALE else 3
### 2.3 CNN image classifier
>filters=64, kernel_size=(3, 3), activation="relu"\
>pool_size=(2, 2)\
>units=64/3, activation="sigmoid"\
>loss="binary_crossentropy", optimizer="Adam", metrics=['accuracy']\
>batch_size=64, epochs=10

## 3 Usage guidelines
### 2.1 Prepare input data
#### Dataset I
>Benchmark dataset: 729 samples from H. sapiens\
>Independent test set: 82 samples form H. sapiens
#### Dataset II
>Benchmark dataset: 219 samples from H. sapiens\
>Independent test set: 65 samples form M. musculus
#### Dataset III
>Benchmark dataset: 505 samples from H. sapiens\
>Independent test set: 805 samples form M. musculus
### 2.2 Generate CGR images in Matalb (2021b)
run generate_CGR_images.m to generate CGR images in batches
>cgr.m is a subfunction that produces a CGR image of a sample
### 2.3 Run CFPLncLoc model in Python (Pycharm 2021)
Run the main program main.py to get the cross-validation and test results
>loaddata.py is a subroutine for loading CGR image features as inputs\
>CFP.py is a subfunction for obtaining CGR image features (based on the multi-scale feature fusion model in computer vision)\
>train.py is a subroutine for CFPLncLoc model training\
>metrics.py is a subroutine for calculating the evaluation metrics AvgF1 and P@1
### 2.4 Interpret the output
After running the model, the results of the following nine evaluation metrics are saved in the results folder:
>MiP, MiR, MiF, MiAUC, MaAUC and HL\
>AP, AvgF1 and P@1
