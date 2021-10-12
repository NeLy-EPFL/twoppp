import numpy as np
import pandas as pd

def get_multi_index_trial_df(trial_info, N_samples, t=None, twop_index=None):
    frames = np.arange(N_samples)
    indices = pd.MultiIndex.from_arrays(([trial_info["Date"], ] * N_samples,  # e.g 210301
                                            [trial_info["Genotype"], ] * N_samples,  # e.g. 'J1xCI9'
                                            [trial_info["Fly"], ] * N_samples,  # e.g. 1
                                            [trial_info["TrialName"], ] * N_samples,  # e.g. 1
                                            [trial_info["i_trial"], ] * N_samples,  # e.g. 1
                                            frames
                                        ),
                                        names=[u'Date', u'Genotype', u'Fly', u'TrialName', u'Trial', u'Frame'])
    df = pd.DataFrame(index=indices)
    if t is not None:
        assert len(t) == N_samples
        df["t"] = t
    if twop_index is not None:
        assert len(twop_index) == N_samples
        df["twop_index"] = twop_index
    return df

def get_multi_index_fly_df(trial_dfs=None):
    multi_index = pd.MultiIndex(levels=[[]] * 5, codes=[[]]  * 5, names=[u'Date', u'Genotype', u'Fly', u'TrialName', u'Trial', u'Frame'])
    df = pd.DataFrame(index=multi_index)
    if trial_dfs is not None and isinstance(trial_dfs, list):
        for trial_df in trial_dfs:
            df.append(trial_df)
    return df

def get_norm_dfs(dfs, perc=0.99):
    mins = []
    maxs = []
    for df in dfs:
        signals = df.filter(regex="neuron").values
        maxs.append(np.quantile(signals, (1+perc)/2, axis=0))
        mins.append(np.quantile(signals, (1-perc)/2, axis=0))
    signal_min = np.mean(mins, axis=0)
    signal_max = np.mean(maxs, axis=0)
    for df in dfs:
        signals = df.filter(regex="neuron").values
        normed_signal = (signals - signal_min) / (signal_max - signal_min)
        N_neurons = normed_signal.shape[1]
        for i_neuron in range(N_neurons):
            df[f"norm_{i_neuron}"] = normed_signal[:,i_neuron]
    return dfs
