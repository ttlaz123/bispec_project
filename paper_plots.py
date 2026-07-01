import numpy as np
import argparse
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.stats import norm
#Local projects
sys.path.insert(0, '/n/home02/toshiyan/Lib/cmblensplus/utils/')
import analysis as ana


def load_experiment_data(bk_type, freq, bn, ell_range, simn, data_path, data_id,
                          jackknife, fiducial_data_path=None, fnl='', calc_gnl=False):
    """
    Loads and preprocesses SIMULATION-ONLY data for a given experiment configuration.
    Real (unblinded) data is intentionally never loaded here.

    Args:
        bk_type (str): Experiment type, e.g., 'BK15' or 'B33y'.
        freq (str): Frequency, e.g., '150' or '100'.
        bn (int): A parameter for file naming.
        ell_range (str): A parameter for the `oL` part of the file path (e.g., 'oL1-300').
        simn (int): Number of simulations.
        data_path (str): Base directory for the data files.
        data_id (int): ID for data extraction from the loaded arrays.

    Returns:
        tuple: (sim_bl, fiducial_bl) arrays.
    """

    print(f"Loading simulation data for experiment: {bk_type}, frequency: {freq}")

    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found at {data_path}")
        raise FileNotFoundError()

    if bk_type == 'B33y':
        tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm{jackknife}_{ell_range}_b{bn}'
        sim_bl_template = os.path.join(data_path, f'{tag}_' + '{}.dat')
        if not fnl == '':
            sim_bl_template = os.path.join(data_path, f'{tag}_' + '{}' + f'_fnl{fnl}.dat')

        nojacktag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm_{ell_range}_b{bn}'
        if fiducial_data_path is None:
            fiducial_data_path = data_path

        template_base = nojacktag.replace("b3d", "mb3d_fnle")
        if calc_gnl:
            # Appends _cubic to find the file generated in the Fortran step
            # e.g., mb3d_fnle..._cubic.dat
            template_base += '_cubic'

        fiducial_bl_path = os.path.join(fiducial_data_path, f'{template_base}.dat')

        if not os.path.exists(fiducial_bl_path):
            print(f"Error: Fiducial file not found: {fiducial_bl_path}")
            raise FileNotFoundError()

        fiducial_bl = np.loadtxt(fiducial_bl_path, unpack=True)
        print(f'Loading sim_bl from template: {sim_bl_template}')
        # Load simulated data with progress
        sim_bl_list = []
        for i in range(simn):
            file_path = sim_bl_template.format(i + 1)  # NOTE: currently starts at sim 1;
                                                          # see message re: reintroducing sim 0
            if i % 50 == 0:
                print(f"Loading simulated file {i + 1}/{simn}: {file_path}")
            txtfile = np.loadtxt(file_path)
            sim_bl_list.append(txtfile.T[data_id, :])
        sim_bl = np.array(sim_bl_list)

    else:
        print(f"Error: Unknown experiment type '{bk_type}'")
        raise FileNotFoundError()

    # Apply the fiducial_bl!=0 filter to sim_bl
    valid_indices = fiducial_bl != 0
    fiducial_bl = fiducial_bl[valid_indices]
    sim_bl = sim_bl[:, valid_indices]

    print(f"Data shapes: sim_bl={np.shape(sim_bl)}, fiducial_bl={np.shape(fiducial_bl)}")
    return sim_bl, fiducial_bl


def compute_covariance_matrix(sim_amplitudes):
    cov = np.cov(sim_amplitudes, rowvar=0)
    cov[np.isnan(cov)] = 0
    return cov


def calc_single_amplitude(relative_amplitude, covmat, diag):
    if diag:
        covmat = np.diag(np.diag(covmat))
    inv_cov = np.linalg.inv(covmat)
    col_sums = np.sum(inv_cov, axis=0)
    total_sum = np.sum(col_sums)
    amplitude = np.sum(col_sums * relative_amplitude) / total_sum
    return amplitude


def run_amplitude_statistics(sim_bl, fiducial_bl, sim_cov_bls=None):
    """
    Computes amplitude statistics (simulation-only) for both diagonal and full
    covariance matrices. No real/observed data is used anywhere in this function.
    """
    fnl_hists = {}
    num_sims = sim_bl.shape[0]

    for diag in [True, False]:
        sim_amplitudes = sim_bl / fiducial_bl

        if sim_cov_bls is None:
            print("No default covariance matrix, calculating from given sims...")
            cov_mat = compute_covariance_matrix(sim_amplitudes)
        else:
            cov_mat = compute_covariance_matrix(sim_cov_bls / fiducial_bl)
        if diag:
            cov_mat = np.diag(np.diag(cov_mat))

        sim_amplitudes_hist = []
        for i in range(num_sims):
            if sim_cov_bls is None:
                leave_one_out_covs = np.cov(np.delete(sim_amplitudes, i, 0), rowvar=0)
            else:
                leave_one_out_covs = cov_mat
            amplitude_i = calc_single_amplitude(sim_amplitudes[i], leave_one_out_covs, diag)
            sim_amplitudes_hist.append(amplitude_i)

        fnl_hists[diag] = sim_amplitudes_hist

    return fnl_hists


def parse_injected_value(fnl_str, is_gnl):
    """
    Converts the zero-padded --fnl string (e.g. '0032') into the numeric
    injected value to draw as the reference line. For gNL plots, converts
    the injected fNL into the corresponding gNL assuming r=1
    (gNL = fNL^3 / r^3 = fNL^3 for r=1).
    """
    if fnl_str == '':
        return None
    injected_fnl = float(fnl_str)
    if is_gnl:
        return injected_fnl ** 3
    return injected_fnl


def plot_fnl_hists(sim_fnl, injected_value, title_name, outpath, param_name="fNL"):
    """
    Plots a (neater) histogram of simulated values with a Gaussian fit overlay.
    The dashed reference line marks the *injected* value (never real/observed data).
    If param_name is 'gNL', also plots the derived fNL = gNL^(1/3) distribution
    side-by-side, with its own Gaussian fit and its own derived injected-value line.
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'figure.facecolor': 'white',
    })

    sim_fnl = np.asarray(sim_fnl)

    if param_name == "gNL":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        ax2 = None

    _plot_single_panel(
        ax1, sim_fnl, injected_value,
        xlabel=f'{param_name} Value',
        hist_color='#6BAED6',
        fit_color='#08519C',
        line_color='#CB181D',
        param_name=param_name,
    )
    ax1.set_title(title_name)

    if param_name == "gNL" and ax2 is not None:
        sim_f_derived = np.cbrt(sim_fnl)
        derived_injected = np.cbrt(injected_value) if injected_value is not None else None

        _plot_single_panel(
            ax2, sim_f_derived, derived_injected,
            xlabel=r'Derived fNL ($g_{NL}^{1/3}$)',
            hist_color='#FC9272',
            fit_color='#A50F15',
            line_color='#67000D',
            param_name='Derived fNL',
        )
        ax2.set_title('Derived fNL Distribution')

    print('Saving: ' + str(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_single_panel(ax, data, injected_value, xlabel, hist_color, fit_color, line_color, param_name):
    """Helper: histogram + Gaussian fit + injected-value line for one axis."""
    sim_mean = np.mean(data)
    sim_std = np.std(data)

    n_bins = 40
    counts, bin_edges, _ = ax.hist(
        data, bins=n_bins, density=False, alpha=0.65,
        color=hist_color, edgecolor='white', linewidth=0.5,
        label=f'Simulated {param_name}\n{sim_mean:.2e} $\\pm$ {sim_std:.2e}'
    )

    # Gaussian fit, scaled to match histogram counts (density=False)
    fit_mu, fit_sigma = norm.fit(data)
    bin_width = bin_edges[1] - bin_edges[0]
    x_smooth = np.linspace(bin_edges[0], bin_edges[-1], 400)
    gaussian_curve = norm.pdf(x_smooth, fit_mu, fit_sigma) * len(data) * bin_width
    ax.plot(
        x_smooth, gaussian_curve, color=fit_color, linewidth=2,
        label=f'Gaussian fit\n$\\mu$={fit_mu:.2e}, $\\sigma$={fit_sigma:.2e}'
    )

    if injected_value is not None:
        ax.axvline(
            injected_value, color=line_color, linestyle='--', linewidth=2,
            label=f'Injected {param_name} = {injected_value:.2e}'
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_amplitudes_to_file(filepath, fnl_sims_dict):
    hists_df = pd.DataFrame(fnl_sims_dict)
    hists_df.columns = ['diag_true', 'diag_false']
    print('Saving amplitudes to csv file: ' + str(filepath))
    hists_df.to_csv(filepath, index_label='sim_index')


def get_amplitudes_from_file(filepath):
    print(f"Loading saved amplitudes from: {filepath}")
    df = pd.read_csv(filepath, index_col='sim_index')
    fnl_hists = {True: df['diag_true'].values, False: df['diag_false'].values}
    return fnl_hists


def main(args):

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    param_label = "gNL" if args.gnl else "fNL"
    injected_value = parse_injected_value(args.fnl, args.gnl)

    for freq in args.freqs:
        tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm{args.jackknife}_dp1102_{args.ell_range}_b{args.bn}'
        # NOTE: 'dp1102' stays in the tag purely for consistent output file naming;
        # no dp1102 (real data) file is ever loaded in this script anymore.

        if args.gnl:
            tag += '_gnl'
        if not args.fnl == '':
            tag = f'{tag}_fnl{args.fnl}'
        amplitudes_file = os.path.join(outdir, tag + '.csv')

        if not os.path.exists(amplitudes_file) or args.overwrite:
            if args.covdata is not None:
                sim_cov_bls, _ = load_experiment_data(
                    args.bk, freq, args.bn, args.ell_range, args.simn, args.covdata,
                    args.id, args.jackknife, fiducial_data_path=args.covdata, fnl='0000')
            else:
                sim_cov_bls = None

            sims_bl, fiducial_bl = load_experiment_data(
                args.bk, freq, args.bn, args.ell_range, args.simn, args.data,
                args.id, args.jackknife, fnl=args.fnl, calc_gnl=args.gnl)

            print(f'Calculating {param_label} amplitudes')
            fnl_hists = run_amplitude_statistics(sims_bl, fiducial_bl, sim_cov_bls)

            save_amplitudes_to_file(amplitudes_file, fnl_hists)
        else:
            fnl_hists = get_amplitudes_from_file(amplitudes_file)

        print(f'Plotting {param_label} Amplitudes')
        for diag in [True, False]:
            title_name = f'{freq}GHz lCDM ell {args.ell_range} with {args.bn} bins and Diag = {diag}'
            if not args.jackknife == '':
                title_name = f'{title_name}, Jackknife={args.jackknife}'
            if not args.fnl == '':
                title_name = f'{title_name}, injected fnl={args.fnl}'

            outpath = os.path.join(outdir, tag + '_diag' + str(diag) + '.png')
            plot_fnl_hists(fnl_hists[diag], injected_value, title_name, outpath, param_name=param_label)