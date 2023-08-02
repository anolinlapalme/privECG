import numpy as np
import argparse
from analyse_output_utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--original", type=str, required=True, help='original ecg numpy array')
    parser.add_argument("-g", "--generated", type=str, required=True, help='generated ecg numpy array')
    args=parser.parse_args()
    return args


def main():
    inputs=parse_args()

    original_array = np.load(inputs.original) 
    original_array = original_array[0:100]

    generated_array = np.load(inputs.generated) 
    original_array = original_array[0:100]
    
    mu_rmse, std_rmse = mean_confidence_interval(rmse(original_array,generated_array))
    mu_frechetdist, std_frechetdist = mean_confidence_interval(frechetdist(original_array,generated_array))
    mu_rho, std_rho = mean_confidence_interval(rho(original_array,generated_array))
    original_std,rmse_std,list_orginal_minus_rmse_total = r_wave_stats(original_array,generated_array)

    mu_original_std, std_original_std = mean_confidence_interval(original_std)
    mu_rmse_std, std_rmse_std = mean_confidence_interval(rmse_std)
    mu_list_orginal_minus_rmse_total, std_list_orginal_minus_rmse_total = mean_confidence_interval(list_orginal_minus_rmse_total)

    print("RMSE µ: {} CI: {}".format(mu_rmse,std_rmse))
    print("Frechet distance µ: {} CI: {}".format(mu_frechetdist,std_frechetdist))
    print("Rho µ: {} CI: {}".format(mu_rho,std_rho))

    print("Variaiton of R peak in original µ: {} CI: {}".format(mu_original_std,std_original_std))
    print("Variaiton of R peak in generated µ: {} CI: {}".format(mu_rmse_std,std_rmse_std))

    print("Difference in R peak aplitude µ: {} CI: {}".format(mu_list_orginal_minus_rmse_total,std_list_orginal_minus_rmse_total))

if __name__ == '__main__':
    main()