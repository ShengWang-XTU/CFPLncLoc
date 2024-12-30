    # **CFPLncLoc**
    A novel deep learning model uses Chaos Game Representation (CGR) images of lncRNA sequences
    and Multi-scale feature fusion (MFF) model Centralized Feature Pyramid (CFP) to predict
    multi-label lncRNA subcellular localization.

    # Requirements
    python==3.8
    torch==1.10.0 or torch==1.13.1+cu117
    torchvision==0.11.1 or torchvision==0.14.1+cu117
    tensorflow==2.2.0 or tensorflow-gpu==2.2.0
    tensorboard==2.2.2
    keras==1.1.2
    pandas==1.1.0
    scipy==1.4.1
    protobuf==3.20.0
    opencv-python==4.10.0.84
    scikit-learn==1.3.2
    scikit-multilearn==0.2.0
    scikit-image==0.18.1
    numpy==1.19.0

    # Setup instructions
    1 Install the dependencies corresponding to the Requirements;
    2 Download pre-trained model (resnet101) from https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    3 Parameter Settings for scatter plot image printing, get three-channel data from image and CNN image classifier.
        A. Print scatter plot image
		pbaspect([1 1 1])                              % Sets the frame aspect ratio to 1:1:1
  		set(gca, 'LooseInset', get(gca, 'TightInset')) % Remove white spaces
	set(gca, 'looseInset', [0 0 0 0])              % Completely remove the interval
	set(gcf, 'color', 'w')                         % Sets the background color to white
        B. Get three-channel data from image
		IMAGE_SIZE = 256  # pixels
  		GRAY_SCALE = False
	CHANNELS = 1 if GRAY_SCALE else 3
        C. CNN image classifier
		filters=64, kernel_size=(3, 3), activation="relu"
	pool_size=(2, 2)
	units=64/3, activation="sigmoid"
	loss="binary_crossentropy", optimizer="Adam", metrics=['accuracy']
		batch_size=64, epochs=10

    # Usage guidelines
    1 Prepare input data
    1.1 Data
        Dataset I
            Benchmark dataset: 729 samples from H. sapiens
            Independent test set: 82 samples form H. sapiens
        Dataset II
            Benchmark dataset: 219 samples from H. sapiens
            Independent test set: 65 samples form M. musculus
        Dataset III
            Benchmark dataset: 505 samples from H. sapiens
            Independent test set: 805 samples form M. musculus
    1.2 Generate CGR images in Matalb (2021b) with generate_CGR_images.m
    1.3 Get the three-channel data of the image in Python (Pycharm 2021) with the following parameter settings:
        IMAGE_SIZE = 256  # pixels
        GRAY_SCALE = False
        CHANNELS = 1 if GRAY_SCALE else 3
    2 Run the model
        Run train_CFP_CGR.py for training and testing in Python
    3 Interpret the output
        After running the model, the following nine metric values are output:
            MiP, MiR, MiF, MiAUC, MaAUC and HL
	    	AP, AvgF1 and P@1
