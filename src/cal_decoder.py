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

from preprocess_eeg import *
from make_dataloader_from_imagelist import make_ds_from_imglist,make_ds_from_veclist
from bigbigan_with_tf_hub import *


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
    x_img_train, x_img_test, y_eeg_train, y_eeg_test = train_test_split(df_imgs.loc[:, 'stim'], df_imgs.loc[:, 'eeg'],
                                                        random_state=config['data_training']['random_state'],
                                                        train_size=config['data_training']['train_rate'])
    return x_img_train, x_img_test, y_eeg_train, y_eeg_test

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
    powers = load_eeglab_timefreq_all(config)
    # extract roi
    powers_roi = extract_roi_for_decoding(powers, config)

    #load image list and connect eeg data to the dataframe
    df_imgs = load_imag_lists_with_eeg(powers_roi, config)

    #for Grootswagers dataset
    df_imgs['stim'] = df_imgs['stim'].str.replace('stimuli', '../stimuli')
    df_imgs['stim'] = df_imgs['stim'].str.replace('\\', '/')

    #make index for the train and test
    x_img_train, x_img_test, y_eeg_train, y_eeg_test = split_train_test_from_df(df_imgs,config)


    #make image loader
    imgs_dataloader = make_ds_from_imglist(list(df_imgs['stim']),
                                           config['dnn']['size_batch'],
                                           (config['dnn']['size_img_w'],config['dnn']['size_img_h']))
    #load pretrained model
    model = load_pretrained_model(config)

    # module = hub.Module(module_path, trainable=True, tags={'train'})  # training
    module = hub.Module(config['dnn']['module_path'])  # inference

    bigbigan = BigBiGAN(module)

    # Make input placeholders for x (`enc_ph`) and z (`gen_ph`).
    enc_ph = bigbigan.make_encoder_ph()
    gen_ph = bigbigan.make_generator_ph()

    # Compute samples G(z) from encoder input z (`gen_ph`).
    gen_samples = bigbigan.generate(gen_ph)

    # Compute reconstructions G(E(x)) of encoder input x (`enc_ph`).
    recon_x = bigbigan.reconstruct_x(enc_ph, upsample=True)

    # Compute encoder features used for representation learning evaluations given
    # encoder input x (`enc_ph`).
    enc_features = bigbigan.encode(enc_ph, return_all_features=True)

    # Compute discriminator scores for encoder pairs (x, E(x)) given x (`enc_ph`)
    # and generator pairs (G(z), z) given z (`gen_ph`).
    disc_scores_enc = bigbigan.discriminate(*bigbigan.enc_pairs_for_disc(enc_ph))
    disc_scores_gen = bigbigan.discriminate(*bigbigan.gen_pairs_for_disc(gen_ph))

    ###################
    # tensor flow initialization
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch = imgs_dataloader[0]
    _out_recons = sess.run(recon_x, feed_dict={enc_ph: batch})
    print('reconstructions shape:', _out_recons.shape)

    inputs_and_recons = interleave(batch, _out_recons)
    print('inputs_and_recons shape:', inputs_and_recons.shape)
    imshow(imgrid(image_to_uint8(inputs_and_recons), cols=2))

    test = 1

    a = 1

