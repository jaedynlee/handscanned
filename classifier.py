import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

import config


def classify(characters):
    """Classify the given characters"""
    try:
        clf = joblib.load(config.CLASSIFIER)
    except FileNotFoundError as fe:
        clf = model()
    predicted = clf.predict(characters)
    return predicted


def model():
    """Create and save an MLPClassifier"""
    # Load dataset. The matrices are already flattened.
    dataset = np.loadtxt(config.DATASET, delimiter=',')
    X = dataset[:, 0:784]
    y = dataset[:, 0]
    # Create mlp classifier
    clf = MLPClassifier()  # TODO parameter tuning
    clf.fit(X, y)
    # Save and return classifier
    joblib.dump(clf, 'clf.pkl')
    return clf
