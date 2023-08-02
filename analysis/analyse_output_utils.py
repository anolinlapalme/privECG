import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
import similaritymeasures
from frechetdist import frdist
from scipy.stats.stats import pearsonr   
from tqdm import tqdm
from ecgdetectors import Detectors
from scipy.signal import find_peaks
import statistics
import adapt

def verify_peaks(known_peaks, scipy_peaks, properties):

    list_peaks_acceptible = list()
    height_of_peak = list()
    for pos,i in enumerate(scipy_peaks):
        for j in known_peaks:
            if i in list(range(j-20, j+21)) and j not in list_peaks_acceptible:
                list_peaks_acceptible.append(j)
                height_of_peak.append(properties[pos])

    return list(zip(list_peaks_acceptible,height_of_peak))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def rmse(original, generated):
    list_rmse = list()
    for original_ECG,generated_ECG in tqdm(zip(original,generated), desc='calculating rmse'):
        for pos_lead, entry_lead in zip(original_ECG,generated_ECG):
            list_rmse.append(mean_squared_error(pos_lead, entry_lead, squared=False))

    return list_rmse


def frechetdist(original, generated):
    list_frechetdist = list()
    for  original_ECG,generated_ECG in tqdm(zip(original,generated), desc='calculating frechet distance'):
        for pos_lead, entry_lead in zip(original_ECG,generated_ECG):

            list_frechetdist.append(adapt.metrics.frechet_distance(pos_lead,entry_lead))

    return list_frechetdist

def rho(original, generated):
    list_rho = list()
    for  original_ECG,generated_ECG in tqdm(zip(original,generated), desc='calculating rho'):
        for pos_lead, entry_lead in zip(original_ECG,generated_ECG):
            
            list_rho.append(pearsonr(pos_lead,entry_lead)[0])

    return list_rho

def make_peaks_unfirom(original, rmse):
    if len(original) < 2 or len(rmse) < 2:
        return 0,0
    pos_1, data_1 = list(zip(*original))
    pos_2, data_2 = list(zip(*rmse))


    pos_in_common = set(pos_1) & set(pos_2)

    original_filterd = [i[1] for i in original if i[0] in pos_in_common]
    rmse_filterd = [i[1] for i in rmse if i[0] in pos_in_common]


    return original_filterd, rmse_filterd




def r_wave_stats(original, generated):


    detectors = Detectors(50)
    original_std = list()
    rmse_std = list()
    list_orginal_minus_rmse_total = list()
    list_orginal_minus_control_total = list()


    for ecg_original, ecg_generated in tqdm(zip(original, generated), desc='calculating R wave stats'):
        r_peaks = detectors.christov_detector(ecg_original[1])

        for lead_original, lead_generated in tqdm(zip(ecg_original, ecg_generated)):

            peaks_original, properties_original = find_peaks(lead_original, prominence=0.02, width=5)
            properties_original = properties_original['prominences']
            tupled_original_ecg = verify_peaks(r_peaks, peaks_original, properties_original)

            peaks_rmse, properties_rmse = find_peaks(lead_generated, prominence=0.02, width=5)
            properties_rmse = properties_rmse['prominences']
            tupled_rmse_ecg = verify_peaks(r_peaks, peaks_rmse, properties_rmse)


            tupled_original_ecg, tupled_rmse_ecg = make_peaks_unfirom(tupled_original_ecg, tupled_rmse_ecg)


            if tupled_original_ecg == 0:
                continue
            #get std and get difference 

            if len(tupled_original_ecg) < 2:
                continue 

            original_std.append(statistics.stdev(tupled_original_ecg))
            rmse_std.append(statistics.stdev(tupled_rmse_ecg))


        difference_original_rmse = np.subtract(tupled_original_ecg,tupled_rmse_ecg)
        list_orginal_minus_rmse_total = [*list_orginal_minus_rmse_total, *difference_original_rmse] 

        return original_std,rmse_std,list_orginal_minus_rmse_total