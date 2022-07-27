import numpy as np
import mne
import io
import pickle
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocess_eeg import *


def extract_roi_for_decoding(powers,start_roi,end_roi):
    return [pow[:, :, start_roi:end_roi].flatten() for pow in powers]

def load_imag_lists(config):
    df_imgs = pd.read_csv(config['fname_imglist'])
    return df_imgs

def split_train_test_from_images(ind_imgs):
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
    powers = load_eeglab_timefreq_all(config['eeg_preprocess'])

    # extract roi
    start_roi=config['data_training']['start_roi']
    end_roi = config['data_training']['start_roi'] + config['data_training']['range_roi']
    powers_roi = extract_roi_for_decoding(powers, start_roi, end_roi)

    #load image list and connect eeg data to the dataframe
    df_imgs = load_imag_lists(config['data_training'])
    if config['eeg_preprocess']['flag_num_powers']==True:
        df_imgs = df_imgs.drop(df_imgs.index[config['eeg_preprocess']['num_powers']:df_imgs.shape[0]+1],axis=0)
    df_imgs['eeg'] = powers_roi

    #make index for the train and test
    x_img_train, x_img_test, y_eeg_train, y_eeg_test = train_test_split(df_imgs.loc[:, 'stim'], df_imgs.loc[:, 'eeg'],
                                                        random_state=config['data_training']['random_state'],
                                                        train_size=config['data_training']['train_rate'])

    #load pretrained model
    model = load_pretrained_model(config)


    a = 1