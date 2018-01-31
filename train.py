"""
Train and export machine learning model using ShipsNet dataset
"""

import sys
import json
import numpy as np
from tflearn.data_utils import to_categorical
from model import model

def train(fname, out_fname):
    """ Train and save CNN model on ShipsNet dataset

    Args:
        fname (str): Path to ShipsNet JSON dataset
        out_fname (str): Path to output Tensorflow model file (.tfl)
    """

    # Load shipsnet data
    f = open(fname)
    shipsnet = json.load(f)
    f.close()

    # Preprocess image data and labels for input
    X = np.array(shipsnet['data']) / 255.
    X = X.reshape([-1,3,80,80]).transpose([0,2,3,1])
    Y = np.array(shipsnet['labels'])
    Y = to_categorical(Y, 2)

    # Train the model
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=.2,
              show_metric=True, batch_size=128, run_id='shipsnet')

    # Save trained model
    model.save(out_fname)


# Main function
if __name__ == "__main__":

    # Train using input file
    train(sys.argv[1], sys.argv[2])
