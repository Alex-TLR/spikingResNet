from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
import time 
import numpy as np

class Metrics():
    def __init__(self, name):
        pass

    @staticmethod
    def softmax(x):
        # Subtract the max value for numerical stability
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def MSP(test_data, threshold):
        '''
        Calculates the Maximum Softmax Probability metrics of input 
        features, regarding the threshold value.

        Inputs:
            test_data:      matrix of input row-wise features
            threshold:      threshold value

        Outputs:
            predictions:    an array of predicted labels
            max_probs:      an array of output features, !-D 
        '''

        s = Metrics.softmax(test_data)
        max_probs = np.max(s, axis=1)
        predictions = (max_probs > threshold).astype(np.int32)
        return predictions, max_probs