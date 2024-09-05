import os
import numpy as np
import scipy
from scipy.signal import hilbert, coherence, correlate, welch
import antropy as ant
import nolds
import pywt
import bisect
from statsmodels.tsa.arima.model import ARIMA
from segment_EEG import segment_EEG
from helper_code import load_recording_data
import pandas as pd
import sys
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import time



MIN_BURST_LEN = 0.5  # Bursts must be at least 0.5s = 500ms per ACNS
MAX_BURST_LEN = 30  # Bursts can be at most 30s per ACNS
MIN_SUPPRESSION_LEN = 1 # Suppressions must be at least 1s = 1000ms per ACNS
MIN_PHASES = 4  # Bursts must have at least 4 phases
SUPPRESSION_THRESHOLD = 10  # Suppression threshold in microvolts (10 uV in this case)

ARMA_AR_MA_NUM = (1, 1)  # Empirically determined optimal (min AIC) number of parameters for ARMA

NUM_BANDS = 5
# Define frequency bands
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, np.inf),
}

BAND_LVLS = {
    0 : 'delta',
    1 : 'theta',
    2 : 'alpha',
    3 : 'beta',
    4 : 'gamma',
}

def burstSuppressionDischarges(x, fs, suppression_threshold=SUPPRESSION_THRESHOLD, bsln_time=30):
    """
    Detect bursts and suppression.
    :param x: signal
    :param fs: sampling rate
    :param suppression_threshold:
    :param bsln_time: time in seconds used to calculate baseline voltage from the beginning of the epoch.
    :return:
    """

    # Calculate envelope
    x_smooth = np.convolve(x, np.ones(10)/10, mode='same') # Smooth using 10-point window
    mag_env = np.abs(np.imag(hilbert(x_smooth)))

    # Calculate baseline
    mag_sorted = np.sort(mag_env)
    baseline = np.mean(mag_sorted[:int(fs * bsln_time)])

    # Remove baseline from envelope -- new baseline is effectively 0
    mag_env = mag_env - baseline

    # Detect suppressions
    under_thresh = (mag_env < suppression_threshold)

    # Remove too-short suppression segments
    supps, suppTimes = filterStateSequence(under_thresh, min_len=int(fs * MIN_SUPPRESSION_LEN))

    # Remove too-short, too-long burst segments
    bursts, burstTimes = filterStateSequence(1 ^ supps, min_len=int(fs * MIN_BURST_LEN),
                                             max_len=int(fs * MAX_BURST_LEN), phase_filt=True, min_phases=4,
                                             sgns=(x_smooth > 0))

    # Anything that is not a burst or suppression
    non_burst_non_supp = (1 ^ supps) & (1 ^ bursts)

    # Discharges are burst-like activity <0.5 s or "bursts" <4 phases
    discharges, dischargeTimes = filterStateSequence(non_burst_non_supp, max_len=int(fs * MAX_BURST_LEN))

    return burstTimes, suppTimes, dischargeTimes


def filterStateSequence(seq, min_len=-np.inf, max_len=np.inf, phase_filt=False, min_phases=0, sgns=None):

    startState = int(seq[0])
    seq = np.append(seq, 1 ^ seq[-1])  # add state opposite to last state to count last state length
    # XOR to find where state changes
    changeInds = seq[1:] ^ seq[:-1]
    stateLens = np.diff(np.nonzero(changeInds)[0], prepend=-1)

    posLens = stateLens[1 ^ startState::2]  # if startState == 1, start from 0
    negLens = stateLens[startState::2]      # if startState == 1, start from 1

    indices = []
    cs = np.insert(np.cumsum(stateLens), 0, 0)

    for i, ps in enumerate(posLens):

        include = True
        s = i * 2 + (1 ^ startState)

        if ps < min_len or ps > max_len:
            seq[cs[s] : cs[s + 1]] = 0
            include = False

        elif phase_filt:
            changes = np.sum(sgns[cs[s] : cs[s + 1] - 1] ^ sgns[cs[s] + 1 : cs[s + 1]])
            if changes <= min_phases:
                seq[cs[s] : cs[s + 1]] = 0
                include= False

        if include:
            indices.append((cs[s], cs[s + 1]))

    return seq[:-1], indices  # remove appended value


def eventStats(events, fs):
    """
    Return mean length and std of events, represented by ndarray of [start, end] rows.
    :param events:
    :return: total, mean, std
    """
    diff = np.diff(events).T.flatten() / fs
    return np.sum(diff), np.mean(diff), np.std(diff)


def discretizeToProb(x, num_bins):

    # Discretize the signal into bins
    histogram, bin_edges = np.histogram(x, bins=num_bins, density=True)

    # Normalize to form a probability distribution
    probabilities = histogram / np.sum(histogram)

    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    return probabilities


def getEntropies(x, num_bins=85, tsallis_qs=10):
    # For fs=256/s, 5 min epochs, n=76800 and 2n^1/3 ~85 per Scott's rule of binning.

    if type(tsallis_qs) is int:
        tsallis_qs = [tsallis_qs]

    probabilities = discretizeToProb(x, num_bins)

    entropies = {}

    entropies['shannon_entropy'] = scipy.stats.entropy(probabilities)

    for q in tsallis_qs:
        q = 1.001 if q == 1 else q  # prevent div by 0
        S = (1 - np.sum([p**q for p in probabilities])) / (q-1)
        entropies['tsallis_q_{0}_entropy'.format(q)] = S

    entropies['approximate_entropy'] = ant.app_entropy(x)
    entropies['permutation_entropy'] = ant.perm_entropy(x, normalize=True)
    entropies['sample_entropy'] = ant.sample_entropy(x)

    return entropies


def subInformationQuality(x, fs):
    """
    Get subband information quality, equivalent to wavelet coefficient entropy.
    :param x: signal
    :param fs: sampling rate
    :return:
    """

    # level i (1 onwards) has a frequency range of Fs/(2^(i-1)) to Fs/(2^i).
    # Here, we want the highest frequency cutoff to be 60, resulting in a 30-60 Hz gamma band; 15-30 beta; 8-15 alpha;
    # 4-8 theta; <4 delta. Note that gamma is defined as 30+ so use lower limit (30) * 2.
    # Note that number of wavelet coefficients are halved at each level.
    # Thus, use n-dependent binning to discretize for entropy calculation.
    num_lvls = NUM_BANDS + int(np.round(np.log2(fs/(BANDS.get('gamma')[0]*2))))
    coefs = pywt.wavedec(x, 'db4', level=num_lvls)
    siq = {}

    for i in range(NUM_BANDS):
        num_bins = int(2 * len(coefs[-i - 1])**(1/3))  # scott's rule
        prbs = discretizeToProb(coefs[-i - 1], num_bins)
        siq[BAND_LVLS.get(i) + "_SIQ"] = scipy.stats.entropy(prbs)

    return siq


def spectral(x, fs, bands=BANDS):
    """
    Spectral power (PSD). Includes mean and median frequency.
    :param x:
    :param fs:
    :return:
    """
    f, psd = welch(x, fs=fs, nperseg=fs*4)

    band_powers = {}

    for b, f_lim in bands.items():
        band_powers[b + "_PSD"] = np.trapz(psd[(f >= f_lim[0]) & (f < f_lim[1])], x=f[(f >= f_lim[0]) & (f < f_lim[1])])

    p_tot = np.sum(psd)
    band_powers['mean_freq'] = np.sum(f * psd) / p_tot

    cum_pow = np.cumsum(psd)
    half_tot = cum_pow[-1] / 2

    # median frequency is the frequency at which the cumulative sum of power reaches half the total power of the signal.
    # Expect this to be close to the middle band of frequencies, thus use binary search to find it.
    band_powers['median_freq'] = f[bisect.bisect_left(cum_pow, half_tot)]

    return band_powers


def arma(x, p, q):

    model = ARIMA(x, order=(p, 0, q))
    res = model.fit()
    return res.arparams, res.maparams


def regularity_cri(x, fs):
    """
    The regularity parameter as defined in the Cerebral Recovery Index.
    Note the parameter is meant to be calculated for a 5 minute-epoch.
    Tjepkema-Cloostermans MC, van Meulen FB, Meinsma G, van Putten MJ. A Cerebral Recovery Index (CRI) for early
    prognosis in patients after cardiac arrest. Crit Care. 2013;17(5):R252. Published 2013 Oct 22.
    doi:10.1186/cc13078
    :param x:
    :return:
    """
    # Step 1: Square the signal
    x_sq = x ** 2

    # Step 2: Create the moving average filter and apply it
    window_len = int(fs * 0.5)  # 500ms window
    wts = np.ones(window_len) / window_len
    x_smooth = np.convolve(x_sq, wts, mode='same')  # Apply filter to signal

    # Step 3: Sort the smoothed signal in descending order
    x_sm_desc = np.sort(x_smooth)[::-1]

    # Step 4: Compute the range statistic
    N = len(x_sm_desc)
    u = np.arange(1, N + 1)
    statsq = np.sum(u ** 2 * x_sm_desc) / (np.sum(x_sm_desc) * (N ** 2) / 3)
    return np.sqrt(statsq)


def twoChCoherence(x1, x2, fs):

    # Calculate the coherence
    f, Cxy = coherence(x1, x2, window='hann', nfft=500, fs=fs)

    band_coherence = {}
    band_coherence['total_coherence'] = np.mean(Cxy)

    # Calculate the mean coherence value in the specified frequency range
    for band, lims in BANDS.items():
        l, h = lims
        band_coherence[f"{band}_coherence"] = np.mean(Cxy[(f >= l) & (f < h)])

    return band_coherence


def twoChMI(x2d):
    discretizer = KBinsDiscretizer(n_bins=200, encode='ordinal', strategy='uniform')
    x_f = discretizer.fit_transform(x2d)
    return mutual_info_score(x_f[0], x_f[1])


def twoChPhaseLag(x1, x2):
    # Calculate the Hilbert transform
    h1 = hilbert(x1)
    h2 = hilbert(x2)

    # Calculate the instantaneous phase
    inst_phasei = np.angle(h1)
    inst_phasej = np.angle(h2)

    # Calculate the output
    return np.abs(np.mean(np.sign(inst_phasej - inst_phasei)))


def crossCorr(x1, x2, fs):
    # Compute cross-correlation
    acor = correlate(x1, x2, mode='full')
    lag = np.arange(-len(x1) + 1, len(x2))

    # Find the index of maximum absolute value of the cross-correlation
    ind = np.argmax(np.abs(acor))

    # Normalized max corr
    maxCorr = (np.max(np.abs(acor)) - np.mean(np.abs(acor))) / np.std(np.abs(acor))
    lag = lag[ind] / fs
    
    return maxCorr, lag


def preprocess(fileName, window_secs=60):

    rescaled_data, channelNames, fs, utility_frequency, st, et = load_recording_data(fileName)
    return *segment_EEG(rescaled_data.astype(np.float64), channelNames, window_time=window_secs,
                        step_time=window_secs//2, Fs=fs, bandpass_freq=(0.5, 50), notch_freq=60), fs, channelNames, \
           st, et


def update_dict_of_dicts(dOfD, outKey, inKey, inVal):

    D = dOfD.get(outKey, dict())
    D[inKey] = inVal
    dOfD[outKey] = D


def calculateFeatures1D(fileName, featuresByTime):
    EEG_segs, BSR_segs, start_ids, seg_masks, specs, freq, fs, chNames, start_time, end_time = preprocess(fileName)

    channels = EEG_segs.transpose((1, 0, 2))  # transpose such that 1st dim in CHANNELS is an electrode channel

    channels_5min = []

    NUM_RAND_SEGS = 3
    common_inds = []

    for num, x in enumerate(channels):

        ch = chNames[num]
        print(num)

        # preprocessing windows were 1 minute long with step size 30 sec.
        # Change new windows to 5 minute epochs, stepsize = 2.5 minutes for overlap.
        full_signal = np.array([])
        for w in x[::2]:
            full_signal = np.append(full_signal, w)

        win_size = int(300 * fs)  # 5 minute window = 300 sec
        step_size = int(win_size // 2)

        sig_windows = []
        i = 0
        while i <= len(full_signal) - win_size:
            sig_windows.append(full_signal[i:i+win_size])
            i = i + step_size


        sig_windows = np.array(sig_windows)
        channels_5min.append(sig_windows)

        b, s, d = burstSuppressionDischarges(full_signal, fs)

        _, burstMean, burstStd = eventStats(b, fs)
        suppTotal, suppMean, suppStd = eventStats(s, fs)
        _, dischMean, dischStd = eventStats(d, fs)

        bsr = fs * suppTotal / len(full_signal)

        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_Burst_len_mean", inKey=start_time[0], inVal=burstMean)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_Burst_len_std", inKey=start_time[0], inVal=burstStd)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_Suppression_len_mean", inKey=start_time[0], inVal=suppMean)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_Suppression_len_std", inKey=start_time[0], inVal=suppStd)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_Discharge_len_mean", inKey=start_time[0], inVal=dischMean)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_Discharge_len_std", inKey=start_time[0], inVal=dischStd)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_Burst_Suppression_Ratio", inKey=start_time[0], inVal=bsr)

        regularity = 0
        entropy_avgs = {}
        bandPow_avgs = {}
        siq_avgs = {}
        arma_ar, arma_ma = 0, 0
        lyapunov = 0
        hjorth_mobility, hjorth_complexity = 0, 0
        hig_fd = 0

        sig_select = sig_windows

        if common_inds and len(sig_windows) > NUM_RAND_SEGS:
            sig_select = sig_windows[common_inds]

        elif len(sig_windows) > NUM_RAND_SEGS:
            inds = [0]
            while len(set(inds)) < NUM_RAND_SEGS:
                inds = (np.random.random(NUM_RAND_SEGS)*len(sig_windows)).astype(int)
            sig_select = sig_windows[inds]
            common_inds = inds

        for signal in sig_select:

            regularity += regularity_cri(signal, fs)

            entropies = getEntropies(signal)
            for k, v in entropies.items():
                entropy_avgs[k] = entropy_avgs.get(k, 0) + v

            bp = spectral(signal, fs)
            for k, v in bp.items():
                bandPow_avgs[k] = bandPow_avgs.get(k, 0) + v

            siq = subInformationQuality(signal, fs)
            for k, v in siq.items():
                siq_avgs[k] = siq_avgs.get(k, 0) + v

            hm, hc = ant.hjorth_params(signal)

            hjorth_mobility += hm
            hjorth_complexity += hc

            hig_fd += ant.higuchi_fd(signal)

        # signal = sig_windows[int(len(sig_windows) * np.random.random())]
        # t1 = time.time()
        # ar, ma = arma(signal, *ARMA_AR_MA_NUM)
        # arma_ar += ar
        # arma_ma += ma
        # t2 = time.time()
        # print(f"arma: {t2-t1}")

        # t1 = time.time()
        # lyapunov += nolds.lyap_r(signal)
        # t2 = time.time()
        # print(f"lyapunov: {t2-t1}")

        denom = len(sig_select)

        regularity /= denom
        hjorth_mobility /= denom
        hjorth_complexity /= denom
        hig_fd /= denom

        for k, v in entropy_avgs.items():
            entropy_avgs[k] = v / denom
            update_dict_of_dicts(featuresByTime, outKey=f"{ch}_{k}", inKey=start_time[0], inVal=v / denom)

        for k, v in bandPow_avgs.items():
            bandPow_avgs[k] = v / denom
            update_dict_of_dicts(featuresByTime, outKey=f"{ch}_{k}", inKey=start_time[0], inVal=v / denom)

        for k, v in siq_avgs.items():
            siq_avgs[k] = v / denom
            update_dict_of_dicts(featuresByTime, outKey=f"{ch}_{k}", inKey=start_time[0], inVal=v / denom)

        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_regularity", inKey=start_time[0], inVal=regularity)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_arma_ar", inKey=start_time[0], inVal=arma_ar)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_arma_ma", inKey=start_time[0], inVal=arma_ma)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_lyapunov", inKey=start_time[0], inVal=lyapunov)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_hjorth_mobility", inKey=start_time[0], inVal=hjorth_mobility)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_hjorth_complexity", inKey=start_time[0], inVal=hjorth_complexity)
        update_dict_of_dicts(featuresByTime, outKey=f"{ch}_hig_fd", inKey=start_time[0], inVal=hig_fd)

    return channels_5min, \
           {"specs": specs, "freq": freq, "fs": fs, "chNames": chNames, "start_time": start_time, "end_time": end_time}


def calculateFeatures2D(X, featuresByTime, spatialFeatures, stime, fs, chNames):

    # coherence
    coherenceByBand = {}
    
    mutual_info = 0
    xcorr, xlag = 0, 0
    phaseLag = 0
    
    pts = 0
    for i in range(len(X)):

        print(chNames[i])

        for j in range(i+1, len(X)):

            print(chNames[j])

            # Find a random 300sec epoch within each 1 hour window
            m = int(len(X[i]) * np.random.random())
            x1 = X[i][m]
            x2 = X[j][m]

            pts += 1
            dict_i_j = dict()

            mi = twoChMI(np.array([x1, x2]))
            mutual_info += mi
            dict_i_j['mutual_information'] = mi

            xc, xl = crossCorr(x1, x2, fs)

            xcorr += xc
            xlag += xl
            dict_i_j['cross_corr'] = xc
            dict_i_j['cross_lag'] = xl

            pl = twoChPhaseLag(x1, x2)
            phaseLag += pl
            dict_i_j['phase_lag'] = pl

            coh = twoChCoherence(x1, x2, fs)
            for k, v in coh.items():
                coherenceByBand[k] = coherenceByBand.get(k, 0) + v
                dict_i_j[k] = v

            spatialFeatures[stime] = spatialFeatures.get(stime, dict())
            d_of_d = spatialFeatures.get(stime)

            for k, v in dict_i_j.items():
                d_of_d[k] = d_of_d.get(k, dict())
                update_dict_of_dicts(d_of_d.get(k),
                                     outKey=chNames[i], inKey=chNames[j], inVal=v)

            #update_dict_of_dicts(spatialFeatures.get(stime), d_of_d)
                    
    mutual_info /= pts
    xcorr /= pts
    xlag /= pts
    phaseLag /= pts

    update_dict_of_dicts(featuresByTime, outKey="Mutual_information", inKey=stime, inVal=mutual_info)
    update_dict_of_dicts(featuresByTime, outKey="Cross_corr_max_norm", inKey=stime, inVal=xcorr)
    update_dict_of_dicts(featuresByTime, outKey="Cross_corr_lag", inKey=stime, inVal=xlag)
    update_dict_of_dicts(featuresByTime, outKey="Phase_lag", inKey=stime, inVal=phaseLag)

    for k, v in coherenceByBand.items():
        coherenceByBand[k] = v / pts
        update_dict_of_dicts(featuresByTime, outKey=k, inKey=stime, inVal=v/pts)


if __name__ == '__main__':

    #eg = scipy.io.loadmat("../EEG_ref/icare/0918/0918_001_003_EEG.mat")
    # fpath meant to contain data from at most 1 patient. pt may be split into multiple paths.
    fpath = sys.argv[1]
    if not fpath.endswith('/'):
        fpath = fpath + "/"
    fileNames = [f'{fpath}{f.replace(".mat", "")}' for f in os.listdir(fpath) if f.endswith('_EEG.mat')]


    ptLevelFeatures = dict()  # dict
    ptLevelSpatial = dict()

    for fileName in fileNames:
        try:
            channels300sec, preprocd = calculateFeatures1D(fileName, ptLevelFeatures)
            st = preprocd.get("start_time")[0]
            fs = preprocd.get("fs")
            chNames = preprocd.get("chNames")
            df = pd.DataFrame.from_dict(ptLevelFeatures, orient='index')
            # Save excel file with just 1D features in case 2D calc fails or times out
            df.to_excel(f"{fpath}features{st}.xlsx")

            calculateFeatures2D(channels300sec, ptLevelFeatures, ptLevelSpatial, st, fs, chNames)
            df = pd.DataFrame.from_dict(ptLevelFeatures, orient='index')
            df.to_excel(f"{fpath}features{st}.xlsx")


        except:
            pass

    for hr, d_of_d in ptLevelSpatial.items():
        for feat2D, vals in d_of_d.items():
            pd.DataFrame.from_dict(vals).to_excel(f"{fpath}twoCh_{hr}_{feat2D}.xlsx")


