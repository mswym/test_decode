import numpy as np
import mne
#follow the tutorial https://mne.tools/dev/auto_tutorials/intro/10_overview.html


def visualize_base(raw, fmax_psd=50, duration=5, n_channels=30):
    #visualize
    raw.plot_psd(fmax=fmax_psd)
    raw.plot(duration=duration, n_channels=n_channels)

def ica_preprocess(raw:object, list_exclude:list, n_component=20, random_state=97, max_iter=800):
    # set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=n_component, random_state=random_state, max_iter=max_iter)
    ica.fit(raw)
    ica.exclude = list_exclude  # details on how we picked these are omitted here
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

def cal_time_frequency(raw,frequencies, n_cycles=2, decim=3, f_visualize=False):
    #frequencies = np.array(7,30,3)
    power = mne.time_frequency.tfr_morlet(raw, n_cycles=n_cycles, return_itc=False,
                                          freqs=frequencies, decim=decim)
    if f_visualize:
        power.plot(0)
    return power

if __name__ == '__main__':
    path_dir = '../data/sub-01/eeg/'
    fname_eeg = 'sub-01_eeg_sub-01_task-rsvp_eeg.vhdr'
    list_exclude = [1]
    frequencies = (5, 30, 5)

    eeg = mne.io.read_raw_brainvision(path_dir + fname_eeg)
    ica = ica_preprocess(eeg, list_exclude)
    eeg = apply_ica(eeg, ica, f_visualize=True)
    power = cal_time_frequency(eeg,frequencies)

    print('finished')