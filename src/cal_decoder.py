import numpy as np
import mne
import io
import pickle
import yaml
import pandas as pd
import matplotlib.pyplot as plt


from preprocess_eeg import *


def extract_roi_for_decoding(powers,start_roi,end_roi):
    return [pow[:, :, start_roi:end_roi].flatten() for pow in powers]

def load_imag_lists(config):
    df_imgs = df.read_csv(config['fname_imglist'])
    return df_imgs

def split_train_test_from_images(path):
    train_images = 1
    test_images = 1
    return train_images, test_images

def load_pretrained_model(config):
    model = 1
    return model


def train_decoder(model,powers_roi,train_images, fname_save):
    decoder = 1

    return decoder

def test_decoder(model, powers, test_images, decoder):
    image_gen = 1
    images_orig = 1

    return images_gen, images_orig

def run_decoding():
    tmp = 1

if __name__ == '__main__':
    #config[0] is for preprocess
    #config[1] is for image and eeg extraction
    path = '../config/config_decoder.yaml'
    with open(path,'r') as f:
        config = yaml.safe_load(f)
    #compute time-frequency info
    powers = load_eeglab_timefreq_all(config[0])

    # extract roi
    start_roi=config[1]['start_roi']
    end_roi = config[1]['start_roi'] + config[1]['range_roi']
    powers_roi = extract_roi_for_decoding(powers, start_roi, end_roi)

    #load image list
    df_imgs = load_imag_lists(config[1])

    #make index for the train and test

    a = 1