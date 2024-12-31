
import numpy as np
from example.train import CFPLncLoc
from loaddata import load

# params.
IMAGE_SIZE = 256  # pixels
SEED = 388014
BATCH_SIZE = 64
EPOCHS = 10
GRAY_SCALE = False
CHANNELS = 1 if GRAY_SCALE else 3
IMAGE_UPDATE = 8  # Image update parameters


# Loading input data
X, y, X_test, y_test = load()
# Training CFPLncLoc model
MiP, MiR, MiF, MiAUC, MaAUC, HL, AP, avgF1, Pat1, mean_y_prob_ho = CFPLncLoc(X, y, X_test, y_test, BATCH_SIZE, EPOCHS)
# Print each evaluation metrics
print("MiP: %.3f MiR: %.3f MiF: %.3f" % (MiP, MiR, MiF))
print("MiAUV: %.3f MaAUC: %.3f HL: %.3f" % (MiAUC, MaAUC, HL))
print("AP: %.3f AvgF1: %.3f P@1: %.3f" % (AP, avgF1, Pat1))
# Splicing test results
results_cv = [MiP, MiR, MiF, MiAUC, MaAUC, HL, AP, avgF1, Pat1]
# Save test results
np.savetxt("test_results.txt", results_cv)
# Save test predicted probabilities
np.savetxt("test_y_prob.txt", mean_y_prob_ho)
