import numpy as np
import mne
import io
import pickle
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub

from sklearn import linear_model

from preprocess_eeg import *
from make_dataloader_from_imagelist import make_ds_from_imglist,make_ds_from_veclist
from bigbigan_with_tf_hub import BigBiGAN



def extract_roi_for_decoding(powers,config):
    start_roi=config['data_training']['start_roi']
    end_roi = config['data_training']['start_roi'] + config['data_training']['range_roi']
    return [pow[:, :, start_roi:end_roi].flatten() for pow in powers]

def load_imag_lists_with_eeg(powers,config):
    df_imgs = pd.read_csv(config['data_training']['fname_imglist'])
    if config['eeg_preprocess']['flag_num_powers']==True:
        df_imgs = df_imgs.drop(df_imgs.index[config['eeg_preprocess']['num_powers']:df_imgs.shape[0]+1],axis=0)
    df_imgs['eeg'] = powers_roi

    return df_imgs

def split_train_test_from_df(df_imgs,config):
    x_eeg_train, x_eeg_test, y_img_train, y_img_test = train_test_split(df_imgs.loc[:, 'eeg'], df_imgs.loc[:, 'stim'],
                                                        random_state=config['data_training']['random_state'],
                                                        train_size=config['data_training']['train_rate'])

    x_eeg_train = [x for x in x_eeg_train]
    x_eeg_train = np.array(x_eeg_train)
    x_eeg_test = [x for x in x_eeg_test]
    x_eeg_test = np.array(x_eeg_test)
    return x_eeg_train, x_eeg_test, y_img_train, y_img_test

def compute_pretrained_model(config,imgs_dataloader):
    # module = hub.Module(module_path, trainable=True, tags={'train'})  # training
    module = hub.Module(config['dnn']['module_path'])  # inference

    bigbigan = BigBiGAN(module)

    # Make input placeholders for x (`enc_ph`) and z (`gen_ph`).
    enc_ph = bigbigan.make_encoder_ph()

    # Compute reconstructions G(E(x)) of encoder input x (`enc_ph`).
    recon_x = bigbigan.reconstruct_x(enc_ph, upsample=True)

    # Compute encoder features used for representation learning evaluations given
    # encoder input x (`enc_ph`).
    enc_features = bigbigan.encode(enc_ph, return_all_features=True)

    ###################
    # tensor flow initialization
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    z_model = []
    for batch in imgs_dataloader:
        #_out_recons = sess.run(recon_x, feed_dict={enc_ph: batch})
        _out_features = sess.run(enc_features, feed_dict={enc_ph: batch})
        z_model.append(_out_features['z_sample'])

    z_model = np.array(z_model)
    return z_model.reshape(z_model.shape[0] * z_model.shape[1], -1)

def train_decoder(x_eeg_train, z_model):
    decoders = []
    for t in range(z_model.shape[1]):
        decoder = linear_model.ARDRegression()
        decoder.fit(x_eeg_train,z_model[:,t].flatten())
        decoders.append(decoder)
    return decoders

def test_decoder(y_eeg_train, z_model):
    image_gen = 1
    images_orig = 1

    return images_gen, images_orig

def run_dataframe_creation(config):
    return a

if __name__ == '__main__':
    #config[0] is for preprocess
    #config[1] is for image and eeg extraction
    path = '../config/config_decoder.yaml'
    with open(path,'r') as f:
        config = yaml.safe_load(f)

    #run_dataframe_creation(config)

    # compute time-frequency info
    powers = load_eeglab_timefreq_all(config)
    # extract roi
    powers_roi = extract_roi_for_decoding(powers, config)

    # load image list and connect eeg data to the dataframe
    df_imgs = load_imag_lists_with_eeg(powers_roi, config)

    # for Grootswagers dataset
    df_imgs['stim'] = df_imgs['stim'].str.replace('stimuli', '../stimuli')
    df_imgs['stim'] = df_imgs['stim'].str.replace('\\', '/')

    # make index for the train and test
    x_eeg_train, x_eeg_test, y_img_train, y_img_test = split_train_test_from_df(df_imgs, config)

    # make image loader
    imgs_dataloader = make_ds_from_imglist(list(y_img_train),
                                           config['dnn']['size_batch'],
                                           (config['dnn']['size_img_w'], config['dnn']['size_img_h']))
    # load pretrained model
    z_model = compute_pretrained_model(config, imgs_dataloader)

    # store z value to dataframe

    a = 1

    # train decoder
    decoder = train_decoder(x_eeg_train, z_model)


    # save dataframe
    df_imgs.to_pickle(config["save"]["fname_save_dataframe"])



    a = 1

