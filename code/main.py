
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
IMAGE_UPDATE = 8


if __name__ == '__main__':
    # Loading input data
    X, y, X_ho, y_ho = load()
    # Training CFPLncLoc model
    AP, HL, avgF1, MiP, MiR, MiF, MaAUC, MiAUC, Pat1, \
    AP_ho, HL_ho, avgF1_ho, MiP_ho, MiR_ho, MiF_ho, MaAUC_ho, \
    MiAUC_ho, Pat1_ho = CFPLncLoc(X, y, X_ho, y_ho, BATCH_SIZE, EPOCHS)
    # Splicing cross-validation results
    results_cv = AP + HL + avgF1 + MiP + MiR + MiF + MaAUC + MiAUC + Pat1
    # Save cross-validation results
    np.savetxt("results/cross_validation_results.txt", results_cv)
    # Print the cross-validation primary result MaAUC
    print("Cross validation MaAUC: %.3f" % MaAUC)

    # Splicing holdout set results
    results_ho = AP_ho + HL_ho + avgF1_ho + MiP_ho + MiR_ho + MiF_ho + MaAUC_ho + MiAUC_ho + Pat1_ho
    # Save holdout set results
    np.savetxt("results/holdout_results.txt", results_ho)
    # Print the holdout set primary result MaAUC
    print("Holdout test MaAUC: %.3f" % MaAUC_ho)
