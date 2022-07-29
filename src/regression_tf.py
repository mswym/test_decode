import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers

print(tf.__version__)

def build_and_compile_model(norm,size_output,learning_rate):
  model = keras.Sequential([
      norm,
      layers.Dense(size_output, activation='linear')
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate))
  return model


if __name__ == '__main__':
    size_output = 120
    learning_rate = 0.1
    epoch = 100
    validation_split = 0.2

    #make normalizer
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features)) #[rep x num_features]

    linear_model = build_and_compile_model(normalizer,size_output,learning_rate)
    history = linear_model.fit(
        train_features,
        train_labels,
        epochs=epoch,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=validation_split)

    with open(fname_file) as f
        pickle.dump(f,(linear_model,history))

