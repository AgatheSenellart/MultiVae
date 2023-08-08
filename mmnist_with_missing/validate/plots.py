"""
Plot the accuracies as a function of the number of input modalities for different models.
"""

import matplotlib.pyplot as plt
import numpy as np
from classifiers import load_mmnist_classifiers

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator
from multivae.models import AutoModel

# One plot per missing ratio

missing_ratios = [0, 0.2, 0.5]
