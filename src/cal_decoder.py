import numpy as np
import mne
import io
import pickle
import yaml

import matplotlib.pyplot as plt
from preprocess_eeg import *


def extract_roi_for_decoding(powers):
    powers_roi = 1

    return powers_roi

def load_pretrained_model(path):
    model = 1

    return model

def load_imag_lists(config):
    df_imgs = df.read_csv(config[1]['fname_imglist'])
    images = 1
    return images

def split_train_test_from_images(path):
    train_images = 1
    test_images = 1
    return train_images, test_images

def train_decoder(model,powers_roi,train_images, fname_save):
    decoder = 1

    return decoder

def test_decoder(model, powers, test_images, decoder):
    image_gen = 1
    images_orig = 1

    return images_gen, images_orig

def run_decoding():

if __name__ == '__main__':
    path = '../config/config_decoder.json'
    with open(path,'r') as f
        config = yaml.safe_load(path)
    powers = load_eeglab_timefreq_all(config[0])