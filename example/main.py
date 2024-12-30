
import numpy as np
from loaddata import load
from train import CFPLncLoc

# params.
IMAGE_SIZE = 256  # pixels
SEED = 388014
BATCH_SIZE = 64
EPOCHS = 10
GRAY_SCALE = False
CHANNELS = 1 if GRAY_SCALE else 3
IMAGE_UPDATE = 8  # Image update parameters

if __name__ == '__main__':
    # Loading input data
    X, y, X_test, y_test = load()
    # Training CFPLncLoc model
    AP, HL, avgF1, MiP, MiR, MiF, MaAUC, MiAUC, Pat1, \
    AP_test, HL_test, avgF1_test, MiP_test, MiR_test, MiF_test, MaAUC_test, \
    MiAUC_test, Pat1_test = CFPLncLoc(X, y, X_test, y_test, BATCH_SIZE, EPOCHS)
    # Splicing cross-validation results
    results_cv = AP + HL + avgF1 + MiP + MiR + MiF + MaAUC + MiAUC + Pat1
    # Save cross-validation results
    np.savetxt("cross_validation_results.txt", results_cv)
    # Print the cross-validation primary result MaAUC
    print("Cross validation MaAUC: %.3f" % MaAUC)

    # Splicing test set results
    results_test = AP_test + HL_test + avgF1_test + MiP_test + MiR_test +\
                   MiF_test + MaAUC_test + MiAUC_test + Pat1_test
    # Save test set results
    np.savetxt("test_results.txt", results_test)
    # Print the test set primary result MaAUC
    print("Independent test MaAUC: %.3f" % MaAUC_test)
