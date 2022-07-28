import numpy as np
import mne
import io
import pickle
import matplotlib.pyplot as plt

import yaml
#follow the tutorial https://mne.tools/dev/auto_tutorials/intro/10_overview.html

def visualize_base(raw, fmax_psd=50, duration=5, n_channels=30):
    #visualize
    raw.plot_psd(fmax=fmax_psd)
    raw.plot(duration=duration, n_channels=n_channels)

def ica_preprocess(raw, list_exclude, n_component=20, random_state=97, max_iter=800, f_visualize=False):
    # set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=n_component, random_state=random_state, max_iter=max_iter)
    ica.fit(raw)
    ica.exclude = list_exclude  # details on how we picked these are omitted here
    if f_visualize:
        ica.plot_properties(raw, picks=ica.exclude)

    return ica

def apply_ica(raw, ica, f_visualize=False):
    orig_raw = raw.copy()
    raw.load_data()
    ica.apply(raw)

    # show some frontal channels to clearly illustrate the artifact removal
    if f_visualize:
        chan_idxs = [0, 1, 2, 3, 4, 5]
        orig_raw.plot(order=chan_idxs, start=12, duration=4)
        raw.plot(order=chan_idxs, start=12, duration=4)

    return raw

def cal_time_frequency(epochs,frequencies, n_cycles=2, decim=3, f_visualize=False):
    #frequencies = np.array(7,30,3)
    power = mne.time_frequency.tfr_morlet(epochs, n_cycles=n_cycles, return_itc=False,
                                          freqs=frequencies, decim=decim)
    if f_visualize:
        power.plot(0)
    return power

def load_eeglab_timefreq_all(config):
    frequencies = config['eeg_preprocess']['frequencies']
    frequencies = np.arange(frequencies[0], frequencies[1], frequencies[2])

    eeglab_raw = mne.io.read_raw_eeglab(config['eeg_preprocess']['path_dir']+config['eeg_preprocess']['fname_eeg'])
    events_from_annot, event_dict = mne.events_from_annotations(eeglab_raw)
    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id=config['eeg_preprocess']['ind_event'],
                        tmin=config['eeg_preprocess']['tmin'], tmax=config['eeg_preprocess']['tmax'],
                        preload=True)

    #get individual epoches
    if config['eeg_preprocess']['flag_num_powers']:
        powers = [cal_time_frequency(epochs.__getitem__(i), frequencies).data for i in range(config['eeg_preprocess']['num_powers'])]
    else:
        powers = [cal_time_frequency(epochs.__getitem__(i), frequencies).data for i in range(epochs.__len__())]
    del epochs, eeglab_raw, events_from_annot

    return powers

if __name__ == '__main__':

    ''' Examples from raw data
    path_dir = '../data/sub-01/eeg/'
    fname_eeg = 'sub-01_eeg_sub-01_task-rsvp_eeg.vhdr'
    list_exclude = [1]
    frequencies = [1, 30, 5]
    #reject_criteria = dict(eeg=150e-6)  # 150 µV
    reject_criteria = dict()
    eeg = mne.io.read_raw_brainvision(path_dir + fname_eeg)
    ica = ica_preprocess(eeg, list_exclude, f_visualize=False)
    eeg = apply_ica(eeg, ica, f_visualize=False)

    events, events_mapping = mne.events_from_annotations(eeg)

    epochs = mne.Epochs(eeg, events, event_id=events_mapping, tmin=-0.2, tmax=0.5,
                        reject=reject_criteria, preload=True)
    del eeg, ica, events, events_mapping
    '''

    ''' 
    Examples from eeg lab data
    '''

    with open('../config/config_decoder.yaml', 'r') as f:
        config = yaml.safe_load(f)

    frequencies = config[0]['frequencies']
    frequencies = np.arange(frequencies[0], frequencies[1], frequencies[2])

    eeglab_raw = mne.io.read_raw_eeglab(config[0]['path_dir']+config[0]['fname_eeg'])
    events_from_annot, event_dict = mne.events_from_annotations(eeglab_raw)
    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id=config[0]['ind_event'],
                        tmin=config[0]['tmin'], tmax=config[0]['tmax'],
                        preload=True)
    #get average data
    #power_mean = cal_time_frequency(epochs, frequencies)
    #get individual epoches
    powers = [cal_time_frequency(epochs.__getitem__(i), frequencies).data for i in range(epochs.__len__())]
    del epochs, eeglab_raw, events_from_annot
    with open(config[0]['fname_save'], 'wb') as f:
        pickle.dump(powers, f, protocol=5)

    #with open('sub-38_power.pickle', 'rb') as f:
    #    powers = pickle.load(f)



    print('finished')