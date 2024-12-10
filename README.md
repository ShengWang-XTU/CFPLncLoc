    # CFPLncLoc
    A novel deep learning model uses Chaos Game Representation (CGR) images of lncRNA sequences and Multi-scale feature fusion (MFF) model Centralized Feature Pyramid (CFP) to predict multi-label lncRNA subcellular localization.

    # Requirements
    python==3.8
    torch==1.10.0
    torchvision==0.11.1
    keras==1.1.2
    numpy==1.20.0
    pandas==1.1.0
    scipy==1.4.1
    tensorflow==2.2.0
    tensorboard==2.2.2
    tqdm==4.66.4
    h5py==2.10.0

    # Setup instructions
    1 Install the dependencies corresponding to the Requirements;
    2 Download pre-trained model (resnet101) from https://download.pytorch.org/models/resnet101-5d3b4d8f.pth

    # Usage guidelines
    1 Prepare input data
    1.1 Data
        Dataset I
            Benchmark dataset: 729 samples from H. sapiens
            Independent test set: 82 samples form H. sapiens
        Dataset II
            Benchmark dataset: 219 samples from H. sapiens
            Independent test set: 65 samples form M. musculus
    1.2 Generate CGR images in Matalb (2021b) with generate_CGR_images.m
    1.3 Get the three-channel data of the image in Python (Pycharm 2021) with the following parameter settings:
        IMAGE_SIZE = 256  # pixels
        GRAY_SCALE = False
        CHANNELS = 1 if GRAY_SCALE else 3
    2 Run the model
        Run train_CFP_CGR.py for training and testing in Python
    3 Interpret the output
        After running the model, the following six metric values are output:
            MiP, MiR, MiF, MiAUC, MaAUC and HL
