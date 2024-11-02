import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import KFold
# from skmultilearn.model_selection import IterativeStratification
from sklearn import metrics as skmetrics
from sklearn.metrics import accuracy_score, precision_score, hamming_loss
from CFP import get_feature


def AvgF1(y_pre, y_true):
    total = 0
    p_total = 0
    p, r = 0, 0
    for yt, yp in zip(y_true, y_pre):
        ytNum = sum(yt)
        if ytNum == 0:
            continue
        rec = sum(yp[yt == 1]) / ytNum
        r += rec
        total += 1
        ypSum = sum(yp)
        if ypSum > 0:
            p_total += 1
            pre = sum(yt[yp == True]) / ypSum
            p += pre
    r /= total
    if p_total > 0:
        p /= p_total
    return 2 * r * p / (r + p)


def PrecisionInTop(Y_prob_pre, Y, n):
    Y_pre = np.argsort(1 - Y_prob_pre, axis=1)[:, :n]
    return sum([sum(y[yp]) for yp, y in zip(Y_pre, Y)]) / (len(Y) * n)


def process_img_file(pp):
    img_arr = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE) if GRAY_SCALE else cv2.imread(str(pp))
    img_arr = img_arr[6:400, 83:477]
    img_arr = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
    return img_arr


def process_X(datapoint: list):
    _x = datapoint.reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    _x = _x / 255.0  # normalize
    return _x


# params.
IMAGE_SIZE = 256  # pixels
SEED = 388014
BATCH_SIZE = 64
EPOCHS = 10
GRAY_SCALE = False
CHANNELS = 1 if GRAY_SCALE else 3

# Read labels 训练集
file = open("data/label_729.csv", "rb")  # Open labels for training set of Dataset I
# file = open("data/label_homo_219.csv", "rb")  # Open labels for training set of Dataset II
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

# Read labels 留出测试集
file = open("data/label_holdout_82.csv", "rb")  # Open labels for holdout test set of Dataset I
# file = open("data/label_mus_65.csv", "rb")  # Open labels for independent test set of Dataset II
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

dd = 256
# 训练集
X = []
for c in range(len(lab)):
    print(c)
    p = "data/CGR_729/CGR_" + str(dd) + "_" + str(c + 1) + ".png"  # Dataset I
    # p = "data/CGR_homo_219/CGR_homo_" + str(dd) + "_" + str(c + 1) + ".png"  # Dataset II
    outs = get_feature(p)
    out = outs['layer1']
    outsq = np.squeeze(out)
    outarr = outsq.detach().numpy()
    outT = outarr.T
    X.append(outT)
X = np.array(X)
y = np.array(labi)

# 留出测试集
X_ho = []
for c in range(len(lab_ho)):
    print(c)
    p = "CGR_holdout_82/CGR_holdout_" + str(dd) + "_" + str(c + 1) + ".png"  # Dataset I
    # p = "data/CGR_mus_65/CGR_mus_" + str(dd) + "_" + str(c + 1) + ".png"  # Dataset II
    outs = get_feature(p)
    out = outs['layer1']
    outsq = np.squeeze(out)
    outarr = outsq.detach().numpy()
    outT = outarr.T
    X_ho.append(outT)
X_ho = np.array(X_ho)
y_ho = np.array(labi_ho)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)  # Dataset I
# kf = IterativeStratification(n_splits=10, order=1)  # Dataset II

one_ACC = []
one_MyACC = []
one_AP = []
one_HL = []
one_avgF1 = []
one_MiP = []
one_MiR = []
one_MiF = []
one_MaAUC = []
one_MiAUC = []
one_Pat1 = []

one_ACC_ho = []
one_MyACC_ho = []
one_AP_ho = []
one_HL_ho = []
one_avgF1_ho = []
one_MiP_ho = []
one_MiR_ho = []
one_MiF_ho = []
one_MaAUC_ho = []
one_MiAUC_ho = []
one_Pat1_ho = []

# 初始化KFold
for train_index, valid_index in kf.split(X):  # 调用split方法切分数据    # Dataset I
    # for train_index, valid_index in kf.split(X, y):  # 调用split方法切分数据    # Dataset II
    X_train, X_val, y_train, y_val = X[train_index], X[valid_index], y[train_index], y[valid_index]
    model = Sequential()
    # input layer: convolution layer of 64 neurons, 3x3 conv window & input shape 60x60x3
    model.add(Conv2D(64, (3, 3), input_shape=X_train[0].shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 pooling window
    # layer 2
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 3
    model.add(Flatten())  # Convert 3D Feature to 1D before Dense layer
    model.add(Dense(64))
    # output layer
    model.add(Dense(4, activation="sigmoid"))  #   4 = no. of labels
    # model.add(Dense(3, activation="sigmoid"))  #   3 = no. of labels
    model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

    # predict batch
    y_prob = model.predict(X_val)
    y_pred = []
    for yi in y_prob:
        y_predi = []
        for yj in yi:
            if yj > 0.5:
                y_predi.append(1)
            else:
                y_predi.append(0)
        y_predi = np.array(y_predi)
        y_pred.append(y_predi)
    y_pred = np.array(y_pred)

    ACC = np.mean(y_pred == y_val)
    MyACC = accuracy_score(y_val, y_pred)
    AP = precision_score(y_val, y_pred, average='samples')
    HL = hamming_loss(y_val, y_pred)
    avgF1 = AvgF1(y_pred, y_val)
    MiP = skmetrics.precision_score(y_val, y_pred, average='micro')
    MiR = skmetrics.recall_score(y_val, y_pred, average='micro')
    MiF = skmetrics.f1_score(y_val, y_pred, average='micro')
    MaAUC = skmetrics.roc_auc_score(y_val, y_prob, average='macro')
    MiAUC = skmetrics.roc_auc_score(y_val, y_prob, average='micro')
    Pat1 = PrecisionInTop(y_prob, y_val, n=1)

    y_prob_ho = model.predict(X_ho)
    y_pred_ho = []
    for yi in y_prob_ho:
        y_predi = []
        for yj in yi:
            if yj > 0.5:
                y_predi.append(1)
            else:
                y_predi.append(0)
        y_predi = np.array(y_predi)
        y_pred_ho.append(y_predi)
    y_pred_ho = np.array(y_pred_ho)

    ACC_ho = np.mean(y_pred_ho == y_ho)
    MyACC_ho = accuracy_score(y_ho, y_pred_ho)
    AP_ho = precision_score(y_ho, y_pred_ho, average='samples')
    HL_ho = hamming_loss(y_ho, y_pred_ho)
    avgF1_ho = AvgF1(y_pred_ho, y_ho)
    MiP_ho = skmetrics.precision_score(y_ho, y_pred_ho, average='micro')
    MiR_ho = skmetrics.recall_score(y_ho, y_pred_ho, average='micro')
    MiF_ho = skmetrics.f1_score(y_ho, y_pred_ho, average='micro')
    MaAUC_ho = skmetrics.roc_auc_score(y_ho, y_prob_ho, average='macro')
    MiAUC_ho = skmetrics.roc_auc_score(y_ho, y_prob_ho, average='micro')
    Pat1_ho = PrecisionInTop(y_prob_ho, y_ho, n=1)

    one_ACC.append(ACC)
    one_MyACC.append(MyACC)
    one_AP.append(AP)
    one_HL.append(HL)
    one_avgF1.append(avgF1)
    one_MiP.append(MiP)
    one_MiR.append(MiR)
    one_MiF.append(MiF)
    one_MaAUC.append(MaAUC)
    one_MiAUC.append(MiAUC)
    one_Pat1.append(Pat1)

    one_ACC_ho.append(ACC_ho)
    one_MyACC_ho.append(MyACC_ho)
    one_AP_ho.append(AP_ho)
    one_HL_ho.append(HL_ho)
    one_avgF1_ho.append(avgF1_ho)
    one_MiP_ho.append(MiP_ho)
    one_MiR_ho.append(MiR_ho)
    one_MiF_ho.append(MiF_ho)
    one_MaAUC_ho.append(MaAUC_ho)
    one_MiAUC_ho.append(MiAUC_ho)
    one_Pat1_ho.append(Pat1_ho)

avg_ACC = np.mean(one_ACC)
avg_MyACC = np.mean(one_MyACC)
avg_AP = np.mean(one_AP)
avg_HL = np.mean(one_HL)
avg_avgF1 = np.mean(one_avgF1)
avg_MiP = np.mean(one_MiP)
avg_MiR = np.mean(one_MiR)
avg_MiF = np.mean(one_MiF)
avg_MaAUC = np.mean(one_MaAUC)
avg_MiAUC = np.mean(one_MiAUC)
avg_Pat1 = np.mean(one_Pat1)

avg_ACC_ho = np.mean(one_ACC_ho)
avg_MyACC_ho = np.mean(one_MyACC_ho)
avg_AP_ho = np.mean(one_AP_ho)
avg_HL_ho = np.mean(one_HL_ho)
avg_avgF1_ho = np.mean(one_avgF1_ho)
avg_MiP_ho = np.mean(one_MiP_ho)
avg_MiR_ho = np.mean(one_MiR_ho)
avg_MiF_ho = np.mean(one_MiF_ho)
avg_MaAUC_ho = np.mean(one_MaAUC_ho)
avg_MiAUC_ho = np.mean(one_MiAUC_ho)
avg_Pat1_ho = np.mean(one_Pat1_ho)

print("5-CV MaAUC: %.3f" % avg_MaAUC)
# print("10-SCV MaAUC: %.3f" % avg_MaAUC)
print("Holdout MaAUC: %.3f" % avg_MaAUC_ho)
# print("Independent MaAUC: %.3f" % avg_MaAUC_ho)
