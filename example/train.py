
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from skmultilearn.model_selection import IterativeStratification
from sklearn import metrics as skmetrics
from sklearn.metrics import precision_score, hamming_loss
from metrics import AvgF1, PrecisionInTop


def CFPLncLoc(X, y, X_ho, y_ho, BATCH_SIZE, EPOCHS):
    """ Training the CFPLncLoc model and obtaining predictive probability and evaluation metrics results """
    one_MiP = []
    one_MiR = []
    one_MiF = []
    one_MiAUC = []
    one_MaAUC = []
    one_HL = []

    one_AP = []
    one_avgF1 = []
    one_Pat1 = []

    one_y_prob = []

    # Stratified 10-fold cross validation
    kf = IterativeStratification(n_splits=10, order=1)
    # initializ KFold
    # Call the split method to split the data
    for train_index, valid_index in kf.split(X, y):
        X_train, X_val, y_train, y_val = X[train_index], X[valid_index], y[train_index], y[valid_index]
        model = Sequential()
        # input layer: convolution layer of 64 neurons, 3x3 conv window
        model.add(Conv2D(64, (3, 3), input_shape=X_train[0].shape, activation="relu"))
        # 2x2 pooling window
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer 2
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer 3
        # Convert 3D Feature to 1D before Dense layer
        model.add(Flatten())
        model.add(Dense(64))
        # output layer
        model.add(Dense(3, activation="sigmoid"))  #   3 = no. of labels
        model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

        # predict batch
        y_prob = model.predict(X_ho)
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

        MiP = skmetrics.precision_score(y_ho, y_pred, average='micro')
        MiR = skmetrics.recall_score(y_ho, y_pred, average='micro')
        MiF = skmetrics.f1_score(y_ho, y_pred, average='micro')
        MiAUC = skmetrics.roc_auc_score(y_ho, y_prob, average='micro')
        MaAUC = skmetrics.roc_auc_score(y_ho, y_prob, average='macro')
        HL = hamming_loss(y_ho, y_pred)

        AP = precision_score(y_ho, y_pred, average='samples')
        avgF1 = AvgF1(y_pred, y_ho)
        Pat1 = PrecisionInTop(y_prob, y_ho, n=1)

        one_MiP.append(MiP)
        one_MiR.append(MiR)
        one_MiF.append(MiF)
        one_MiAUC.append(MiAUC)
        one_MaAUC.append(MaAUC)
        one_HL.append(HL)

        one_AP.append(AP)
        one_avgF1.append(avgF1)
        one_Pat1.append(Pat1)

        one_y_prob.append(y_prob)

    avg_MiP = np.mean(one_MiP)
    avg_MiR = np.mean(one_MiR)
    avg_MiF = np.mean(one_MiF)
    avg_MiAUC = np.mean(one_MiAUC)
    avg_MaAUC = np.mean(one_MaAUC)
    avg_HL = np.mean(one_HL)

    avg_AP = np.mean(one_AP)
    avg_avgF1 = np.mean(one_avgF1)
    avg_Pat1 = np.mean(one_Pat1)

    # Calculate the average result of the predicted probabilities
    mean_y_prob_ho = np.mean(one_y_prob, axis=0)

    return avg_MiP, avg_MiR, avg_MiF, avg_MiAUC, avg_MaAUC, avg_HL, avg_AP, avg_avgF1, avg_Pat1, mean_y_prob_ho
