import numpy as np
import argparse
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
#Local projects
sys.path.insert(0, '/n/home02/toshiyan/Lib/cmblensplus/utils/')
import analysis as ana


def load_experiment_data(bk_type, freq, bn, ell_range, simn, data_path, data_id, jackknife, fiducial_data_path=None, fnl='', calc_gnl=False):
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
        tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm{jackknife}_{ell_range}_b{bn}'
        observed_bl_path = os.path.join(data_path, f'{tag.replace("_oL", "_dp1102_oL")}_0.dat')
        sim_bl_template = os.path.join(data_path, f'{tag}_' + '{}.dat')
        if(not fnl == ''):
            sim_bl_template = os.path.join(data_path, f'{tag}_' + '{}' + f'_fnl{fnl}.dat')

        nojacktag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm_{ell_range}_b{bn}'
        if(fiducial_data_path is None):
            fiducial_data_path = data_path

        template_base = nojacktag.replace("b3d", "mb3d_fnle")
        if calc_gnl:
            # Appends _cubic to find the file generated in the Fortran step
            # e.g., mb3d_fnle..._cubic.dat
            template_base += '_cubic'
            
        fiducial_bl_path = os.path.join(fiducial_data_path, f'{template_base}.dat')

# Check if files exist
        if not os.path.exists(observed_bl_path):
            observed_bl_path = os.path.join(data_path, f'{tag.replace("_oL", "_dp1102_oL")}_0_fnl{fnl}.dat')
        if not os.path.exists(observed_bl_path):
            print(f"Error: Observed file not found: {observed_bl_path}")
            raise FileNotFoundError()

        if not os.path.exists(fiducial_bl_path):
            print(f"Error: Fiducial file not found: {fiducial_bl_path}")
            raise FileNotFoundError()

            
        observed_bl = np.loadtxt(observed_bl_path).T
        if(data_id == -1):
            data_id = observed_bl.shape[0]-1
        observed_bl = observed_bl[data_id, :]
        print('Loading fiducial_bl:' + fiducial_bl_path)
        fiducial_bl = np.loadtxt(fiducial_bl_path, unpack=True)
        print(f'Loading sim_bl from template: {sim_bl_template}')
        # Load simulated data with progress
        sim_bl_list = []
        for i in range(simn):
            file_path = sim_bl_template.format(i + 1)
            if(i % 50 == 0):
                print(f"Loading simulated file {i + 1}/{simn}: {file_path}")
            txtfile = np.loadtxt(file_path)
            sim_bl_list.append(txtfile.T[data_id, :])
        sim_bl = np.array(sim_bl_list)
        
        
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

def calc_single_amplitude_bruteforce(fiducial_bl, covmat, sim_bl, diag):
    if(diag):
        covmat = np.diag(np.diag(covmat))
    inv_cov = np.linalg.inv(covmat)
    total_sum = np.sum(inv_cov)
    fnl = np.matmul(np.matmul(fiducial_bl, inv_cov), sim_bl)
    return fnl

def plot_covmat(mat):
    nonzeros = np.abs(mat[(mat!=0) &( ~np.isnan(mat))])
    vpercent = np.percentile(nonzeros, 99)
    vmax = np.max(nonzeros)
    linthresh = np.percentile(nonzeros, 1)
    log_mags = np.logspace(np.floor(np.log10(linthresh)), 
                        np.ceil(np.log10(vmax)), num=10, base=10)
    custom_ticks = np.concatenate([
                 -log_mags[::-1],  # Negative ticks (reversed)
                [0.0],            # Zero tick
                log_mags          # Positive ticks
            ])
    cmap = plt.get_cmap('seismic')
    norm = mcolors.SymLogNorm(linthresh=linthresh,
                            vmin=-vpercent,vmax = vpercent, base=10)
    plt.figure()
    #plt.imshow(cov_mat_inv, cmap=cmap, vmin=-vpercent, vmax=vpercent)
    plt.imshow(mat,cmap=cmap, norm = norm)
    plt.title('Inverse covar matrix')
    plt.xlabel('bispec bin ijk')
    plt.ylabel('bispec bin ijk')
    plt.colorbar(ticks=custom_ticks, format='%.1e')
    #plt.colorbar()
    plt.savefig('invcovmat.png')

def run_amplitude_statistics(observed_bl, sim_bl, fiducial_bl, sim_cov_bls = None):
    """
    Computes amplitude statistics for both diagonal and full covariance matrices.
    """
    fnl_hists = {}
    fnl_obs = {}
    num_sims = sim_bl.shape[0]
    brute_force = False
    for diag in [True, False]:
         
        sim_amplitudes = sim_bl/fiducial_bl
        if(brute_force):
            cov_mat = compute_covariance_matrix(sim_cov_bls)
            if(diag):
                cov_mat = np.diag(np.diag(cov_mat))
            avg_sim_bls = np.mean(sim_cov_bls, axis=0)
            cov_mat_inv = np.linalg.inv(cov_mat)
            plot_covmat(cov_mat_inv)
            denom = np.matmul(np.matmul(avg_sim_bls, cov_mat_inv), avg_sim_bls)

        elif(sim_cov_bls is None):
            cov_mat = compute_covariance_matrix(sim_amplitudes)
        else:
            cov_mat = compute_covariance_matrix(sim_cov_bls/fiducial_bl)
        if(diag):
            cov_mat = np.diag(np.diag(cov_mat))
        if(brute_force):
            numer = np.matmul(np.matmul(observed_bl, cov_mat_inv),fiducial_bl)
            observed_amplitude = numer/denom
        else:
            observed_amplitude = calc_single_amplitude(observed_bl/fiducial_bl,
                                            cov_mat, diag)
        fnl_obs[diag] = (observed_amplitude)
        sim_amplitudes_hist = []
        for i in range(num_sims):
            if(sim_cov_bls is None and not brute_force): 
                leave_one_out_covs = np.cov(np.delete(sim_amplitudes, i, 0), 
                                                rowvar=0)
            else:
                leave_one_out_covs = cov_mat
            if(brute_force):
                numer = np.matmul(np.matmul(avg_sim_bls, cov_mat_inv),sim_bl[i])
                amplitude_i = numer/denom

            else:
                amplitude_i = calc_single_amplitude(sim_amplitudes[i],
                                        leave_one_out_covs, diag)
            sim_amplitudes_hist.append(amplitude_i)

        fnl_hists[diag] = (sim_amplitudes_hist)
        
    return fnl_hists, fnl_obs

def plot_fnl_hists(sim_fnl, obs_fnl, title_name, outpath, param_name="fNL"):

    """
    Plots a histogram of simulated fNL values and a vertical line for the observed value.

    Args:
        sim_fnl (np.ndarray): An array of simulated fNL values.
        obs_fnl (float): The single observed fNL value.
    """
    plt.figure(figsize=(10, 6))
    sim_mean = np.mean(sim_fnl)
    sim_std = np.std(sim_fnl)
    '''
    if(sim_std > 10):
        bin_range = 5000
    else:
        bin_range = 5
    bin_width = bin_range/20
    bins = np.arange(-bin_range, bin_range+bin_width, bin_width)
    
    if(sim_std > 5000 or sim_std < 0.01):
        bins = None
    else:    
        print('Clipping outliers for visualization')
        sim_fnl = np.clip(sim_fnl, -bin_range,bin_range)
    '''
    bins=40
    plt.hist(sim_fnl, bins=bins, density=False, alpha=0.6, color='skyblue', 
             label=f'Simulated {param_name} ' + str(np.round(sim_mean,6)) + '+-' + str(np.round(sim_std,6)))
    
    plt.axvline(obs_fnl, color='red', linestyle='dashed', linewidth=2, 
                label=f'Observed {param_name} = '+ str(np.round(obs_fnl,6)))
    
    plt.xlabel(f'{param_name} Value')
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
    
    outdir = args.outdir
    if(not os.path.exists(outdir)):
        os.mkdir(outdir)

    # Set parameter name string for plots
    param_label = "gNL" if args.gnl else "fNL"
    for freq in args.freqs:
        tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm{args.jackknife}_dp1102_{args.ell_range}_b{args.bn}'
        
        # Modify tag so gNL results don't overwrite fNL results
        if args.gnl:
            tag += '_gnl'
        if(not args.fnl ==''):
            tag = f'{tag}_fnl{args.fnl}'
        amplitudes_file = os.path.join(outdir, tag + '.csv')
        if(not os.path.exists(amplitudes_file) or args.overwrite):
            # Load and preprocess the data
            if(not args.covdata is None):
                observed_bl, sim_cov_bls, fiducial_bl = load_experiment_data(args.bk, freq, 
                                args.bn, args.ell_range, args.simn, args.covdata, args.id, args.jackknife, 
                                fiducial_data_path=args.covdata, fnl=args.fnl, calc_gnl=args.gnl)
            else:
                sim_cov_bls = None
            observed_bl, sims_bl, fiducial_bl = load_experiment_data(args.bk, freq, 
                                args.bn, args.ell_range, args.simn, args.data, args.id, args.jackknife, 
                                fnl=args.fnl, calc_gnl=args.gnl)
            
            # Run the amplitude statistics calculation
            print(f'Calculating {param_label} amplitudes')
            fnl_hists, fnl_obs = run_amplitude_statistics(observed_bl, sims_bl, 
                                                            fiducial_bl, sim_cov_bls)
            
            save_amplitudes_to_file(amplitudes_file, fnl_hists, fnl_obs)
        else:
            fnl_hists, fnl_obs  = get_amplitudes_from_file(amplitudes_file)
        print(f'Plotting {param_label} Amplitudes')
        for diag in [True, False]:
            title_name = f'{freq}GHz lCDM ell {args.ell_range} with {args.bn} bins and Diag = {diag}'
            if(not args.jackknife ==''):
                title_name = f'{title_name}, Jackknife={args.jackknife}'
            if(not args.fnl == ''):
                title_name = f'{title_name}, injected fnl={args.fnl}'
            
            outpath = os.path.join(outdir, tag + '_diag' + str(diag) + '.png')
            
            # Pass the param_name to plotting
            plot_fnl_hists(fnl_hists[diag], fnl_obs[diag], title_name, outpath, param_name=param_label)

def ensure_underscores(s):
    """
    Ensures a string is surrounded by underscores.
    
    Args:
        s (str): The input string.

    Returns:
        str: The string with underscores added if they were missing.
    """
    if(not s):
        return ""
    if not s.startswith('_'):
        s = '_' + s
    return s

if __name__ == "__main__":
    """
    Main entry point for the bispec_analysis script.
    
    Parses command-line arguments and runs the analysis.
    """
    parser = argparse.ArgumentParser(description='Calculate bispectrum statistics.')
    parser.add_argument('--bn', type=int, default=10, help='Bin number parameter.')
    parser.add_argument('--bk', type=str, default='B33y', help='Experiment type (e.g., BK15 or B33y).')
    parser.add_argument('--id', type=int, default=-1, help='ID column for data extraction.')
    parser.add_argument('--freqs', nargs='+', default=['100'], help='List of frequencies to analyze.')
    parser.add_argument('--simn', type=int, default=499, help='Number of simulations.')
    
    parser.add_argument('--ell_range', type=str, default='oL1-300', 
                help='The ell_range parameter for file names (e.g., oL1-300, 20-350, 20-580, 30-350).')
    parser.add_argument('--jackknife', type=str, default='',
                            help='Jackknife type')
    parser.add_argument('--data', type=str, 
                default='/n/holylfs06/LABS/kovac_lab/users/liuto/B33y/bispec_bbb/', 
                help='Base data directory.')
    parser.add_argument('--covdata', type=str, 
                default='/n/holylfs06/LABS/kovac_lab/users/namikawa/B33y/bispec_bbb/', 
                help='Base data directory for calculating covariance matrix.')

    parser.add_argument('-o', '--overwrite', action='store_true',
                    help = 'whether to overwrite the csv file')
    parser.add_argument('--fnl', default='', 
                    help='fnl value: 0001, 0010, 0100, 0032, 1000')
    parser.add_argument('--gnl', action='store_true', help='Calculate gNL instead of fNL')

    
    args = parser.parse_args()
    if(args.data == 'liuto'):
        args.data = '/n/holylfs06/LABS/kovac_lab/users/liuto/B33y/bispec_bbb/'
    elif(args.data == 'namikawa'):
        args.data = '/n/holylfs06/LABS/kovac_lab/users/namikawa/B33y/bispec_bbb/'
    if('liuto' in args.data):
        args.outdir = 'fnl_figs_scaled2'    
    elif('namikawa' in args.data):
        args.outdir = 'fnl_figs_test'
    if(args.covdata=='None'):
        args.covdata = None
        args.outdir='fnl_figs_defcov'
    if(args.covdata == 'liuto'):
        args.covdata = '/n/holylfs06/LABS/kovac_lab/users/liuto/B33y/bispec_bbb/'

    for ellrange in ['oL20-350']:#,'oL30-350']:#, 'oL20-580']:#, 'oL30-350']:
        for bin_num in [9]:#,8,9]:#[7,8,9, 10, 11,16]:
            for jack in ['']:#,'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'ja', 'jb', 'jc', 'jd', 'je']:
                
                args.bn = bin_num
                args.ell_range = ellrange
                
                try:
                    args.jackknife=ensure_underscores(jack)
                    main(args)
                except FileNotFoundError:
                    print('Combo does not exist: ' + ellrange + ' bn' + str(bin_num))
                    pass


