
import numpy as np
from CFP import get_feature


# params.
IMAGE_SIZE = 256  # pixels
SEED = 388014
BATCH_SIZE = 64
EPOCHS = 10
GRAY_SCALE = False
CHANNELS = 1 if GRAY_SCALE else 3
IMAGE_UPDATE = 8


def load():
    """ Obtaining and loading input features for CFPLncLoc models """
    # Read training set labels
    file = open("data/label_homo_219.csv", "rb")  # Open labels for training set
    lab = np.loadtxt(file, delimiter=',', skiprows=0)
    file.close()
    labi = []
    for i in lab:
        labj = []
        for j in i:
            labj.append(int(j))
        labj = np.array(labj)
        labi.append(labj)
    labi = np.array(labi)

    # Read independent test set labels
    file = open("data/label_mus_65.csv", "rb")  # Open labels for independent test set
    lab_ho = np.loadtxt(file, delimiter=',', skiprows=0)
    file.close()
    labi_ho = []
    for i in lab_ho:
        labj = []
        for j in i:
            labj.append(int(j))
        labj = np.array(labj)
        labi_ho.append(labj)
    labi_ho = np.array(labi_ho)

    # Load image input for training set
    X = []
    for c in range(len(lab)):
        print(f"{c} / {len(lab)}")
        p = "data/CGR_homo_219/CGR_homo_" + str(pow(2, IMAGE_UPDATE)) + "_" + str(c + 1) + ".png"
        outs = get_feature(p)
        out = outs['layer1']
        outsq = np.squeeze(out)
        outarr = outsq.detach().numpy()
        outT = outarr.T
        X.append(outT)
    X = np.array(X)
    y = np.array(labi)

    # Load image input for independent test set
    X_ho = []
    for c in range(len(lab_ho)):
        print(f"{c} / {len(lab_ho)}")
        p = "data/CGR_mus_65/CGR_mus_" + str(pow(2, IMAGE_UPDATE)) + "_" + str(c + 1) + ".png"
        outs = get_feature(p)
        out = outs['layer1']
        outsq = np.squeeze(out)
        outarr = outsq.detach().numpy()
        outT = outarr.T
        X_ho.append(outT)
    X_ho = np.array(X_ho)
    y_ho = np.array(labi_ho)
    return X, y, X_ho, y_ho


if __name__ == '__main__':
    load()
