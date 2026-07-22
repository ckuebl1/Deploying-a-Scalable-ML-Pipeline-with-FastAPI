import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import apply_label
from ml.model import train_model, compute_model_metrics


def test_apply_labels():
    """
    Test that apply_label correctly converts a binary prediction into the expected string.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns expected values when predictions match true labels. 
    """
    true_labels = np.array([1, 0, 1, 1, 0])
    predicted_labels = np.array([1, 0, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(true_labels, predicted_labels)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_train_model():
    """
    Test that train_model returns a trained RandomForestClassifier instance.
    """
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
