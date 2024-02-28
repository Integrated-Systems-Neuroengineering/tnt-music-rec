# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from deepthought_master.deepthought.datasets.openmiir.preprocessing.pipeline import Pipeline 


subject = 'P01' # TODO: change this for each subject
verbose = True
settings = dict(debug=False, mne_log_level='Info', sfreq=64) # optional pipeline settings
pipeline = Pipeline(subject, settings)

raw = mne.io.read_raw_fif('P01-raw.fif', allow_maxshield=False, preload=True, on_split_missing='raise', verbose=None)
#raw.plot()

#raw.info.set_montage('standard_1020', match_case=False, on_missing='warn')

#montage = raw.info.get_montage()
#mne.viz.plot_montage(montage)


raw_filt = raw.copy().filter(l_freq = 1, h_freq = 50)
raw_filt.compute_psd().plot()

raw_filt.plot()

raw_cropped = raw.crop(tmin=0, tmax=3).load_data()

# ocular artifacts
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine="mean")
eog_epochs.average().plot_joint()

# bad blinks programatically
eog_events = mne.preprocessing.find_eog_events(raw)
onsets = eog_events[:, 0] / raw.info["sfreq"] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ["bad blink"] * len(eog_events)
blink_annot = mne.Annotations(
    onsets, durations, descriptions, orig_time=raw.info["meas_date"]
)
raw.set_annotations(blink_annot)
eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
raw.plot(events=eog_events, order=eeg_picks)

eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

#ICA components 
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica

explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )
    
explained_var_ratio = ica.get_explained_variance_ratio(
    filt_raw, components=[0], ch_type="eeg"
)
# This time, print as percentage.
ratio_percent = round(100 * explained_var_ratio["eeg"])
print(
    f"Fraction of variance in EEG signal explained by first component: "
    f"{ratio_percent}%"
)
raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)

# blinks
ica.plot_overlay(raw, exclude=[0], picks="eeg")
ica.plot_components()

