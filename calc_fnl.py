import numpy as np
import argparse
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
#Local projects
sys.path.insert(0, '/n/home02/toshiyan/Lib/cmblensplus/utils/')
import analysis as ana


def load_experiment_data(bk_type, freq, bn, ell_range, simn, data_path, data_id):
    """
    Loads and preprocesses data for a given experiment configuration.
    
    Args:
        bk_type (str): Experiment type, e.g., 'BK15' or 'B33y'.
        freq (str): Frequency, e.g., '150' or '100'.
        bn (int): A parameter for file naming.
        ell_range (str): A parameter for the `oL` part of the file path (e.g., 'oL1-300').
        simn (int): Number of simulations.
        data_path (str): Base directory for the data files.
        data_id (int): ID for data extraction from the loaded arrays.

    Returns:
        tuple: (observed_bl, sim_bl, fiducial_bl) arrays.
    """
    
    
    
    print(f"Loading data for experiment: {bk_type}, frequency: {freq}")

    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found at {data_path}")
        raise FileNotFoundError()

    if bk_type == 'B33y':
        tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm_{ell_range}_b{bn}'

        observed_bl_path = os.path.join(data_path, f'{tag.replace("cdm", "cdm_Vdst")}_1.dat')
        sim_bl_template = os.path.join(data_path, f'{tag}_' + '{}.dat')
        fiducial_bl_path = os.path.join(data_path, f'{tag.replace("b3d", "mb3d_fnle")}.dat')

# Check if files exist
        if not os.path.exists(observed_bl_path):
            print(f"Error: Observed file not found: {observed_bl_path}")
            raise FileNotFoundError()

        if not os.path.exists(fiducial_bl_path):
            print(f"Error: Fiducial file not found: {fiducial_bl_path}")
            raise FileNotFoundError()

            
        observed_bl = np.loadtxt(observed_bl_path).T[data_id, :]
        print(f'Loading sim_bl from template: {sim_bl_template}')
        # Load simulated data with progress
        sim_bl_list = []
        for i in range(simn):
            file_path = sim_bl_template.format(i + 1)
            if(i % 50 == 0):
                print(f"Loading simulated file {i + 1}/{simn}: {file_path}")
            sim_bl_list.append(np.loadtxt(file_path).T[data_id, :])
        sim_bl = np.array(sim_bl_list)
        print('Loading fiducial_bl')
        fiducial_bl = np.loadtxt(fiducial_bl_path, unpack=True)
        
    else:
        print(f"Error: Unknown experiment type '{bk_type}'")
        raise FileNotFoundError()

    # Apply the fiducial_bl!=0 filter to sim_bl and observed_bl
    valid_indices = fiducial_bl != 0
    if observed_bl is not None:
        observed_bl = observed_bl[valid_indices]
    fiducial_bl = fiducial_bl[valid_indices]
    sim_bl = sim_bl[:, valid_indices]

    print(f"Data shapes: sim_bl={np.shape(sim_bl)}, fiducial_bl={np.shape(fiducial_bl)}")
    return observed_bl, sim_bl, fiducial_bl

def compute_covariance_matrix(sim_amplitudes):
    cov = np.cov(sim_amplitudes, rowvar = 0)
    cov[np.isnan(cov)] = 0
    return cov

def calc_single_amplitude(relative_amplitude, covmat, diag):
    if(diag):
        covmat = np.diag(np.diag(covmat))
    inv_cov = np.linalg.inv(covmat)
    col_sums = np.sum(inv_cov, axis = 0)
    total_sum = np.sum(col_sums)
    amplitude = np.sum(col_sums * relative_amplitude) / total_sum
    return amplitude

def run_amplitude_statistics(observed_bl, sim_bl, fiducial_bl):
    """
    Computes amplitude statistics for both diagonal and full covariance matrices.
    """
    fnl_hists = {}
    fnl_obs = {}
    num_sims = sim_bl.shape[0]
    for diag in [True, False]:
        sim_amplitudes = sim_bl/fiducial_bl
        cov_mat = compute_covariance_matrix(sim_amplitudes)
        observed_amplitude = calc_single_amplitude(observed_bl/fiducial_bl,
                                            cov_mat, diag)
        fnl_obs[diag] = (observed_amplitude)
        sim_amplitudes_hist = []
        for i in range(num_sims):
            
            leave_one_out_covs = np.cov(np.delete(sim_amplitudes, i, 0), 
                                                rowvar=0)
            amplitude_i = calc_single_amplitude(sim_amplitudes[i],
                                        leave_one_out_covs, diag)
            sim_amplitudes_hist.append(amplitude_i)

        fnl_hists[diag] = (sim_amplitudes_hist)
        
    return fnl_hists, fnl_obs

def plot_fnl_hists(sim_fnl, obs_fnl, title_name, outpath):

    """
    Plots a histogram of simulated fNL values and a vertical line for the observed value.

    Args:
        sim_fnl (np.ndarray): An array of simulated fNL values.
        obs_fnl (float): The single observed fNL value.
    """
    plt.figure(figsize=(10, 6))
    bins = np.arange(-5000, 5000+250, 250)
    
    sim_mean = np.mean(sim_fnl)
    sim_std = np.std(sim_fnl)
    
    print('Clipping outliers for visualization')
    sim_fnl = np.clip(sim_fnl, -5000,5000)
    plt.hist(sim_fnl, bins=bins, density=False, alpha=0.6, color='skyblue', label='Simulated fNL ' + str(np.round(sim_mean,3)) + '+-' + str(np.round(sim_std,3)))
    
    # Plot the observed fNL value as a vertical red line.
    plt.axvline(obs_fnl, color='red', linestyle='dashed', linewidth=2, label='Observed fNL = '+ str(np.round(obs_fnl,3)))
    
    # Add labels, a title, and a legend to make the plot clear.
    plt.xlabel('fNL Value')
    plt.ylabel('Count')
    plt.title(title_name)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    print('Saving: ' + str(outpath))
    plt.savefig(outpath)
    # Display the plot

    return 
def save_amplitudes_to_file(filepath, fnl_sims_dict, fnl_obs_dict):
    hists_df = pd.DataFrame(fnl_sims_dict)
    hists_df.columns = ['diag_true', 'diag_false']
    # Create a DataFrame for the observed data, with a specific index
    obs_df = pd.DataFrame([fnl_obs_dict], index=['observed'])
    obs_df.columns = ['diag_true', 'diag_false']
    combined_df = pd.concat([obs_df, hists_df])
    print('Saving amplitudes to csv file: ' + str(filepath))
    combined_df.to_csv(filepath, index_label='type')

def get_amplitudes_from_file(filepath):
    print(f"Loading save amplitudes from: {filepath}")
    df = pd.read_csv(filepath, index_col='type')
        
    # Extract the observed row and convert it to a dictionary
    obs_row = df.loc['observed'].to_dict()
    fnl_obs = {True: obs_row['diag_true'], False: obs_row['diag_false']}
        
    # Extract the simulated data (all other rows) and convert to dict of arrays
    hists_df = df.drop('observed')
    fnl_hists = {True: hists_df['diag_true'].values, False: hists_df['diag_false'].values}
    return fnl_hists, fnl_obs

def main(args):
    
    outdir = 'fnl_figs'
    if(not os.path.exists(outdir)):
        os.mkdir(outdir)
    for freq in args.freqs:
        tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm_{args.ell_range}_b{args.bn}'
        amplitudes_file = os.path.join(outdir, tag + '.csv')
        if(not os.path.exists(amplitudes_file)):
            # Load and preprocess the data
            observed_bl, sims_bl, fiducial_bl = load_experiment_data(args.bk, freq, 
                                                args.bn, args.ell_range, args.simn, args.data, args.id)
        
            # Run the amplitude statistics calculation
            print('Calculating amplitudes')
            fnl_hists, fnl_obs = run_amplitude_statistics(observed_bl, sims_bl, fiducial_bl)
            save_amplitudes_to_file(amplitudes_file, fnl_hists, fnl_obs)
        else:
            fnl_hists, fnl_obs  = get_amplitudes_from_file(amplitudes_file)
        print('Plotting Amplitudes')
        for diag in [True, False]:
            title_name = f'{freq}GHz lCDM ell {args.ell_range} with {args.bn} bins and Diag = ' + str(diag)
            
            outpath = os.path.join(outdir, tag + '_diag' + str(diag) + '.png')
            plot_fnl_hists(fnl_hists[diag], fnl_obs[diag], title_name, outpath)


if __name__ == "__main__":
    """
    Main entry point for the bispec_analysis script.
    
    Parses command-line arguments and runs the analysis.
    """
    parser = argparse.ArgumentParser(description='Calculate bispectrum statistics.')
    parser.add_argument('--bn', type=int, default=10, help='Bin number parameter.')
    parser.add_argument('--bk', type=str, default='B33y', help='Experiment type (e.g., BK15 or B33y).')
    parser.add_argument('--id', type=int, default=3, help='ID column for data extraction.')
    parser.add_argument('--freqs', nargs='+', default=['100'], help='List of frequencies to analyze.')
    parser.add_argument('--simn', type=int, default=499, help='Number of simulations.')
    
    parser.add_argument('--ell_range', type=str, default='oL1-300', 
                help='The ell_range parameter for file names (e.g., oL1-300, 20-350, 20-580, 30-350).')
    
    parser.add_argument('--data', type=str, 
                default='/n/holylfs04/LABS/kovac_lab/users/namikawa/B33y/bispec_bbb/', 
                help='Base data directory.')

    
    args = parser.parse_args()
    for ellrange in ['oL1-300', 'oL20-350', 'oL20-580', 'oL30-350']:
        for bin_num in [7,8,9, 10, 11,16]:
            args.bn = bin_num
            args.ell_range = ellrange
            try:
                main(args)
            except FileNotFoundError:
                print('Combo does not exist: ' + ellrange + ' bn' + str(bin_num))
                pass


