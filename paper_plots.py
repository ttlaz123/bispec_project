print('Importing directories')
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob
import re
print('Done importing')
BASE_USERS = {
    'liuto': '/n/holylfs06/LABS/kovac_lab/users/liuto/B33y/',
    'namikawa': '/n/holylfs06/LABS/kovac_lab/users/namikawa/B33y/',
}
NOISE_DIRS = {
    'with': 'bispec_bbb_almflmglm',    # confirmed: includes noise
    'without': 'bispec_bbb_flmglm',    # confirmed: no noise
}


# ---------------------------------------------------------------------------
# Data loading (simulation-only; no real/unblinded data is ever read here)
# ---------------------------------------------------------------------------


def find_sim_files(data_path, tag, fnl):
    """
    Finds all simulation files for a given tag by globbing the directory,
    rather than assuming any particular index range or starting value.
    Real data is excluded structurally: sim filenames never contain a
    dp-tag (e.g. 'dp1102'), so a plain glob on the no-dp tag naturally
    can't match real-data files, but we double-check explicitly anyway.
    """
    if fnl != '':
        pattern = os.path.join(data_path, f'{tag}_*_fnl{fnl}.dat')
    else:
        pattern = os.path.join(data_path, f'{tag}_*.dat')

    matches = glob.glob(pattern)

    # Defense in depth: explicitly drop anything with a dp-tag, in case a
    # directory's naming convention ever changes.
    dp_pattern = re.compile(r'_dp\d+_')
    sim_files = [f for f in matches if not dp_pattern.search(os.path.basename(f))]

    # Extract the index (the integer between the tag and the optional _fnl suffix)
    index_pattern = re.compile(re.escape(tag) + r'_(\d+)(?:_fnl\d+)?\.dat$')
    indexed = []
    for f in sim_files:
        m = index_pattern.search(os.path.basename(f))
        if m:
            indexed.append((int(m.group(1)), f))
    indexed.sort(key=lambda x: x[0])

    if not indexed:
        raise FileNotFoundError(f"No simulation files found matching pattern: {pattern}")

    indices = [i for i, _ in indexed]
    print(f"Found {len(indexed)} sim files for tag '{tag}', "
          f"index range {min(indices)}-{max(indices)} "
          f"({'contiguous' if indices == list(range(min(indices), max(indices)+1)) else 'NON-CONTIGUOUS, check for gaps'})")

    return indexed
def resolve_path(path, noise):
    """Expand 'liuto'/'namikawa' shorthand into a full noise-aware directory path."""
    if path in BASE_USERS:
        return os.path.join(BASE_USERS[path], NOISE_DIRS[noise])
    return path

def load_experiment_data(bk_type, freq, bn, ell_range, simn, data_path, data_id,
                          jackknife, fiducial_data_path=None, fnl='', calc_gnl=False):
    """
    Loads SIMULATION-ONLY bispectrum data. Files are discovered by globbing
    the directory rather than assuming any index range; any file containing
    a dp-tag (e.g. dp1102) is real data and is explicitly excluded.
    """
    print(f"Loading simulation data for experiment: {bk_type}, frequency: {freq}")

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found at {data_path}")
    if bk_type != 'B33y':
        raise FileNotFoundError(f"Unknown experiment type '{bk_type}'")

    tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm{jackknife}_{ell_range}_b{bn}'

    nojacktag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm_{ell_range}_b{bn}'
    if fiducial_data_path is None:
        fiducial_data_path = data_path
    template_base = nojacktag.replace("b3d", "mb3d_fnle")
    if calc_gnl:
        template_base += '_cubic'
    fiducial_bl_path = os.path.join(fiducial_data_path, f'{template_base}.dat')
    if not os.path.exists(fiducial_bl_path):
        raise FileNotFoundError(f"Fiducial file not found: {fiducial_bl_path}")
    fiducial_bl = np.loadtxt(fiducial_bl_path, unpack=True)

    indexed_files = find_sim_files(data_path, tag, fnl)

    if simn is not None and len(indexed_files) < simn:
        raise FileNotFoundError(
            f"Requested simn={simn} but only found {len(indexed_files)} sim files "
            f"for tag '{tag}' (fnl='{fnl}') in {data_path}")
    if simn is not None:
        indexed_files = indexed_files[:simn]  # take the first `simn` by index order

    first_idx, first_path = indexed_files[0]
    first_arr = np.loadtxt(first_path).T
    if data_id == -1:
        data_id = first_arr.shape[0] - 1

    sim_bl_list = [first_arr[data_id, :]]
    for n, (idx, path) in enumerate(indexed_files[1:], start=1):
        if n % 50 == 0:
            print(f"Loading simulated file {n}/{len(indexed_files)}: {path}")
        sim_bl_list.append(np.loadtxt(path).T[data_id, :])
    sim_bl = np.array(sim_bl_list)

    valid_indices = fiducial_bl != 0
    fiducial_bl = fiducial_bl[valid_indices]
    sim_bl = sim_bl[:, valid_indices]

    print(f"Data shapes: sim_bl={np.shape(sim_bl)}, fiducial_bl={np.shape(fiducial_bl)}")
    return sim_bl, fiducial_bl


# ---------------------------------------------------------------------------
# Amplitude statistics
# ---------------------------------------------------------------------------

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
    return np.sum(col_sums * relative_amplitude) / total_sum


def run_amplitude_statistics(sim_bl, fiducial_bl, sim_cov_bls=None):
    """Simulation-only amplitude recovery for both diagonal and full covariance."""
    fnl_hists = {}
    num_sims = sim_bl.shape[0]

    for diag in [True, False]:
        sim_amplitudes = sim_bl / fiducial_bl

        if sim_cov_bls is None:
            print("No external covariance sims given; computing from analysis sims...")
            cov_mat = compute_covariance_matrix(sim_amplitudes)
        else:
            cov_mat = compute_covariance_matrix(sim_cov_bls / fiducial_bl)
        if diag:
            cov_mat = np.diag(np.diag(cov_mat))

        sim_amplitudes_hist = []
        for i in range(num_sims):
            if sim_cov_bls is None:
                leave_one_out_cov = np.cov(np.delete(sim_amplitudes, i, 0), rowvar=0)
            else:
                leave_one_out_cov = cov_mat
            sim_amplitudes_hist.append(
                calc_single_amplitude(sim_amplitudes[i], leave_one_out_cov, diag)
            )
        fnl_hists[diag] = sim_amplitudes_hist

    return fnl_hists


def parse_injected_value(fnl_str, is_gnl):
    """Numeric injected value for the reference line. gNL = fNL^3 (r=1 only)."""
    if fnl_str == '':
        return None
    injected_fnl = float(fnl_str)
    return injected_fnl ** 3 if is_gnl else injected_fnl


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_single_panel(ax, data, injected_value, xlabel, hist_color, fit_color, line_color, param_name):
    data = np.asarray(data)
    sim_mean, sim_std = np.mean(data), np.std(data)

    n_bins = 40
    counts, bin_edges, _ = ax.hist(
        data, bins=n_bins, alpha=0.65, color=hist_color, edgecolor='white', linewidth=0.5,
        label=f'Simulated {param_name}\n{sim_mean:.2e} $\\pm$ {sim_std:.2e}'
    )

    fit_mu, fit_sigma = norm.fit(data)
    bin_width = bin_edges[1] - bin_edges[0]
    x_smooth = np.linspace(bin_edges[0], bin_edges[-1], 400)
    ax.plot(
        x_smooth, norm.pdf(x_smooth, fit_mu, fit_sigma) * len(data) * bin_width,
        color=fit_color, linewidth=2,
        label=f'Gaussian fit\n$\\mu$={fit_mu:.2e}, $\\sigma$={fit_sigma:.2e}'
    )

    if injected_value is not None:
        ax.axvline(injected_value, color=line_color, linestyle='--', linewidth=2,
                   label=f'Injected {param_name} = {injected_value:.2e}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_fnl_hists(sim_fnl, injected_value, title_name, outpath, param_name="fNL"):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                          'axes.titleweight': 'bold', 'figure.facecolor': 'white'})

    if param_name == "gNL":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        ax2 = None

    _plot_single_panel(ax1, sim_fnl, injected_value, f'{param_name} Value',
                        '#6BAED6', '#08519C', '#CB181D', param_name)
    ax1.set_title(title_name)

    if ax2 is not None:
        sim_f_derived = np.cbrt(sim_fnl)
        derived_injected = np.cbrt(injected_value) if injected_value is not None else None
        _plot_single_panel(ax2, sim_f_derived, derived_injected, r'Derived fNL ($g_{NL}^{1/3}$)',
                            '#FC9272', '#A50F15', '#67000D', 'Derived fNL')
        ax2.set_title('Derived fNL Distribution')

    print('Saving: ' + str(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_amplitudes_to_file(filepath, fnl_sims_dict):
    df = pd.DataFrame(fnl_sims_dict)
    df.columns = ['diag_true', 'diag_false']
    print('Saving amplitudes to: ' + str(filepath))
    df.to_csv(filepath, index_label='sim_index')


def get_amplitudes_from_file(filepath):
    print(f"Loading cached amplitudes from: {filepath}")
    df = pd.read_csv(filepath, index_col='sim_index')
    return {True: df['diag_true'].values, False: df['diag_false'].values}


def ensure_underscores(s):
    if not s:
        return ""
    return s if s.startswith('_') else '_' + s


# ---------------------------------------------------------------------------
# Single-configuration run
# ---------------------------------------------------------------------------

def run_one_config(args, freq, bn, ell_range, jackknife):
    param_label = "gNL" if args.gnl else "fNL"
    injected_value = parse_injected_value(args.fnl, args.gnl)

    # Tag now encodes noise setting and data source explicitly, so different
    # --noise / --data runs can never collide or silently overwrite each other.
    tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm{jackknife}_{ell_range}_b{bn}_{args.noise}noise'
    if args.gnl:
        tag += '_gnl'
    if args.fnl != '':
        tag += f'_fnl{args.fnl}'

    amplitudes_file = os.path.join(args.outdir, tag + '.csv')

    if not os.path.exists(amplitudes_file) or args.overwrite:
        sim_cov_bls = None
        if args.covdata is not None:
            sim_cov_bls, _ = load_experiment_data(
                args.bk, freq, bn, ell_range, args.simn, args.covdata,
                args.id, jackknife, fiducial_data_path=args.covdata, fnl='0000')

        sims_bl, fiducial_bl = load_experiment_data(
            args.bk, freq, bn, ell_range, args.simn, args.data,
            args.id, jackknife, fnl=args.fnl, calc_gnl=args.gnl)

        print(f'Calculating {param_label} amplitudes')
        fnl_hists = run_amplitude_statistics(sims_bl, fiducial_bl, sim_cov_bls)
        save_amplitudes_to_file(amplitudes_file, fnl_hists)
    else:
        fnl_hists = get_amplitudes_from_file(amplitudes_file)

    print(f'Plotting {param_label} amplitudes')
    for diag in [True, False]:
        title_bits = [f'{freq}GHz lCDM ell {ell_range}, {bn} bins, diag={diag}',
                      f'noise={args.noise}']
        if jackknife:
            title_bits.append(f'jackknife={jackknife}')
        if args.fnl != '':
            title_bits.append(f'injected fnl={args.fnl}')
        title_name = ', '.join(title_bits)

        outpath = os.path.join(args.outdir, f'{tag}_diag{diag}.png')
        plot_fnl_hists(fnl_hists[diag], injected_value, title_name, outpath, param_name=param_label)
# ---------------------------------------------------------------------------
# Full scan mode: cubic vs linear × noise vs no-noise × all fnl values
# ---------------------------------------------------------------------------

ALL_FNL_VALUES = ['0000', '0001', '0010', '0032', '0100', '1000']


def run_scan(args, freq, bn, ell_range, jackknife):
    """
    Runs every combination of estimator (linear/cubic) x noise (with/without)
    x fnl value, collecting fitted Gaussian parameters into one table instead
    of just producing plots. Skips combos with missing files rather than
    aborting the whole scan.
    """
    results = []

    for gnl in [False, True]:
        estimator_label = 'cubic' if gnl else 'linear'
        for noise in ['with', 'without']:
            data_path = resolve_path('liuto', noise)
            covdata_path = None if args.covdata is None else resolve_path(args.covdata, noise)

            for fnl in ALL_FNL_VALUES:
                injected_value = parse_injected_value(fnl, gnl)

                try:
                    sim_cov_bls = None
                    if covdata_path is not None:
                        sim_cov_bls, _ = load_experiment_data(
                            args.bk, freq, bn, ell_range, args.simn, covdata_path,
                            args.id, jackknife, fiducial_data_path=covdata_path, fnl='0000')

                    sims_bl, fiducial_bl = load_experiment_data(
                        args.bk, freq, bn, ell_range, args.simn, data_path,
                        args.id, jackknife, fnl=fnl, calc_gnl=gnl)

                    fnl_hists = run_amplitude_statistics(sims_bl, fiducial_bl, sim_cov_bls)

                except FileNotFoundError as e:
                    print(f'  [skip] estimator={estimator_label} noise={noise} fnl={fnl}: {e}')
                    for diag in [True, False]:
                        results.append({
                            'estimator': estimator_label, 'noise': noise, 'injected_fnl': fnl,
                            'diag': diag, 'injected_value': injected_value,
                            'fit_mu': None, 'fit_sigma': None, 'n_sims': None, 'status': 'missing_files',
                        })
                    continue

                for diag in [True, False]:
                    data = np.asarray(fnl_hists[diag])
                    fit_mu, fit_sigma = norm.fit(data)
                    results.append({
                        'estimator': estimator_label, 'noise': noise, 'injected_fnl': fnl,
                        'diag': diag, 'injected_value': injected_value,
                        'fit_mu': fit_mu, 'fit_sigma': fit_sigma,
                        'n_sims': len(data), 'status': 'ok',
                    })
                    print(f'  estimator={estimator_label:6s} noise={noise:7s} fnl={fnl} diag={diag}: '
                          f'mu={fit_mu:.4e}  sigma={fit_sigma:.4e}  (n={len(data)})')

                    # Also make the plot while we're here, same as run_one_config
                    tag = (f'b3d_1var_nu{freq}_lx0_nobl_lcdm{jackknife}_{ell_range}_b{bn}'
                           f'_{noise}noise{"_gnl" if gnl else ""}_fnl{fnl}')
                    outpath = os.path.join(args.outdir, f'{tag}_diag{diag}.png')
                    title_name = (f'{freq}GHz lCDM ell {ell_range}, {bn} bins, diag={diag}, '
                                  f'noise={noise}, injected fnl={fnl}')
                    plot_fnl_hists(data, injected_value, title_name, outpath,
                                    param_name='gNL' if gnl else 'fNL')

    return pd.DataFrame(results)


def print_scan_table(df):
    """Prints a clean, copy-pasteable table of fitted Gaussian parameters."""
    ok = df[df['status'] == 'ok'].copy()
    ok['fit_mu'] = ok['fit_mu'].map(lambda x: f'{x:.4e}')
    ok['fit_sigma'] = ok['fit_sigma'].map(lambda x: f'{x:.4e}')
    ok['injected_value'] = ok['injected_value'].map(lambda x: f'{x:.4e}' if x is not None else '')

    cols = ['estimator', 'noise', 'injected_fnl', 'injected_value', 'diag', 'fit_mu', 'fit_sigma', 'n_sims']
    print("\n" + "=" * 100)
    print("FITTED GAUSSIAN PARAMETERS (copy-paste table)")
    print("=" * 100)
    print(ok[cols].to_string(index=False))
    print("=" * 100)

    missing = df[df['status'] == 'missing_files']
    if len(missing):
        print(f"\n{len(missing)} combo(s) skipped due to missing files:")
        for _, row in missing.iterrows():
            print(f"  estimator={row['estimator']} noise={row['noise']} fnl={row['injected_fnl']} diag={row['diag']}")

# ---------------------------------------------------------------------------
# CLI
#def build_parser():
    p = argparse.ArgumentParser(description='Calculate and plot bispectrum fNL/gNL statistics (simulation-only).')
    p.add_argument('--bk', type=str, default='B33y')
    p.add_argument('--id', type=int, default=-1, help='Row index (-1 = last row).')
    p.add_argument('--freqs', nargs='+', default=['100'])
    p.add_argument('--simn', type=int, default=0,
                   help='Number of sims to use (first N found, sorted by index). 0 = use all found.')

    p.add_argument('--ell_ranges', nargs='+', default=['oL20-350'])
    p.add_argument('--bins', nargs='+', type=int, default=[9])
    p.add_argument('--jackknifes', nargs='+', default=[''])

    p.add_argument('--data', type=str, default='liuto',
                   help="Data directory, or shorthand 'liuto' / 'namikawa'.")
    p.add_argument('--covdata', type=str, default='None',
                   help="Covariance directory, shorthand, or 'None' for leave-one-out.")
    p.add_argument('--noise', choices=['with', 'without'], default='with',
                   help="Ignored when --scan-all is set (scan covers both).")

    p.add_argument('--fnl', default='', help="Ignored when --scan-all is set (scan covers all fnl values).")
    p.add_argument('--gnl', action='store_true', help="Ignored when --scan-all is set (scan covers both).")
    p.add_argument('--scan-all', action='store_true',
                   help='Run every combo of estimator x noise x fnl in one pass and print a summary table.')
    p.add_argument('--table-csv', default=None,
                   help='If set (with --scan-all), also writes the summary table to this CSV path.')

    p.add_argument('-o', '--overwrite', action='store_true')
    p.add_argument('--outdir', default=None)
    return p


def resolve_args(args):
    if not args.scan_all:
        args.data = resolve_path(args.data, args.noise)
        if args.covdata == 'None':
            args.covdata = None
        else:
            args.covdata = resolve_path(args.covdata, args.noise)

    if args.outdir is None:
        args.outdir = 'scan_figs' if args.scan_all else f'{"gnl" if args.gnl else "fnl"}_figs_{args.noise}noise'
    os.makedirs(args.outdir, exist_ok=True)

    if args.simn == 0:
        args.simn = None

    return args


def main():
    args = resolve_args(build_parser().parse_args())

    for freq in args.freqs:
        for bn in args.bins:
            for ell_range in args.ell_ranges:
                for jack in args.jackknifes:
                    jackknife = ensure_underscores(jack)
                    if args.scan_all:
                        print(f"\n--- Scanning freq={freq} bn={bn} ell_range={ell_range} jack={jack} ---")
                        df = run_scan(args, freq, bn, ell_range, jackknife)
                        print_scan_table(df)
                        if args.table_csv:
                            df.to_csv(args.table_csv, index=False)
                            print(f"\nFull results also written to: {args.table_csv}")
                    else:
                        try:
                            run_one_config(args, freq, bn, ell_range, jackknife)
                        except FileNotFoundError as e:
                            print(f'Skipping combo (freq={freq}, bn={bn}, ell_range={ell_range}, '
                                  f'jack={jack}): {e}')


if __name__ == "__main__":
    main()