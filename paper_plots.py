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


def compute_summary_stats(data, injected_value=None):
    """Gaussian fit + percentile-based stats, plus bias relative to the
    injected value when available. For skewed distributions (notably
    gNL = fNL^3), the median/percentile stats are the honest summary;
    the Gaussian fit is kept for comparison but is a poor description
    of gNL's shape."""
    fit_mu, fit_sigma = norm.fit(data)
    median = np.median(data)
    p16, p84 = np.percentile(data, [16, 84])
    stats = {
        'fit_mu': fit_mu, 'fit_sigma': fit_sigma,
        'median': median, 'p16': p16, 'p84': p84,
        'lower_1sigma_eq': median - p16, 'upper_1sigma_eq': p84 - median,
    }
    if injected_value is not None:
        stats['bias_mu'] = fit_mu - injected_value
        stats['bias_median'] = median - injected_value
    return stats


def calc_single_amplitude(relative_amplitude, covmat, diag):
    if diag:
        covmat = np.diag(np.diag(covmat))
    inv_cov = np.linalg.inv(covmat)
    col_sums = np.sum(inv_cov, axis=0)
    total_sum = np.sum(col_sums)
    return np.sum(col_sums * relative_amplitude) / total_sum


def run_amplitude_statistics(sim_bl, fiducial_bl, fixed_cov_amplitudes=None):
    """
    Simulation-only amplitude recovery for both diagonal and full covariance.

    fixed_cov_amplitudes:
        - None: self-referential leave-one-out covariance is used (correct
          ONLY when sim_bl is itself the null/fnl=0000 set, to avoid a sim
          contributing to the covariance used to evaluate itself).
        - an (Nbin x Nbin) amplitude covariance matrix: used as-is for every
          sim. This is the normal case for any fnl != 0000, where the
          covariance comes from the independent null sims and there's no
          self-referencing to guard against.
    """
    fnl_hists = {}
    num_sims = sim_bl.shape[0]
    sim_amplitudes = sim_bl / fiducial_bl

    for diag in [True, False]:
        sim_amplitudes_hist = []
        for i in range(num_sims):
            if fixed_cov_amplitudes is None:
                cov_mat = np.cov(np.delete(sim_amplitudes, i, 0), rowvar=0)
            else:
                cov_mat = fixed_cov_amplitudes
            sim_amplitudes_hist.append(
                calc_single_amplitude(sim_amplitudes[i], cov_mat, diag)
            )
        fnl_hists[diag] = sim_amplitudes_hist

    return fnl_hists


def parse_injected_value(fnl_str, is_gnl):
    if fnl_str == '':
        return None
    injected_fnl = float(fnl_str)
    return injected_fnl ** 3 if is_gnl else injected_fnl


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_single_panel(ax, data, injected_value, xlabel, hist_color, fit_color,
                        line_color, param_name, center_on_injected=False):
    data = np.asarray(data)

    if center_on_injected and injected_value is not None:
        data = data - injected_value
        plotted_injected = 0.0
        # xlabel intentionally left unchanged (still just "fNL"/"gNL"),
        # per request — the shift is implicit in the plot, not the label.
    else:
        plotted_injected = injected_value

    fit_mu, fit_sigma = norm.fit(data)
    median = np.median(data)
    p16, p84 = np.percentile(data, [16, 84])

    n_bins = 40
    counts, bin_edges, _ = ax.hist(
        data, bins=n_bins, alpha=0.65, color=hist_color, edgecolor='white', linewidth=0.5,
        label=f'Simulated {param_name}\nGaussian: {fit_mu:.2e} $\\pm$ {fit_sigma:.2e}\n'
              f'Median: {median:.2e} [{p16:.2e}, {p84:.2e}]'
    )

    bin_width = bin_edges[1] - bin_edges[0]
    x_smooth = np.linspace(bin_edges[0], bin_edges[-1], 400)
    ax.plot(x_smooth, norm.pdf(x_smooth, fit_mu, fit_sigma) * len(data) * bin_width,
            color=fit_color, linewidth=2, label='Gaussian fit')

    ax.axvline(median, color=fit_color, linestyle=':', linewidth=1.5, label='Median')
    ax.axvspan(p16, p84, color=fit_color, alpha=0.12, label='16th-84th percentile')

    if plotted_injected is not None:
        ax.axvline(plotted_injected, color=line_color, linestyle='--', linewidth=2,
                   label=f'Injected {param_name} = {plotted_injected:.2e}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_fnl_hists(sim_fnl, injected_value, title_name, outpath, param_name="fNL",
                    center_on_injected=False):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                          'axes.titleweight': 'bold', 'figure.facecolor': 'white'})

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    _plot_single_panel(ax1, sim_fnl, injected_value, param_name,   # <- plain label
                        '#6BAED6', '#08519C', '#CB181D', param_name,
                        center_on_injected=center_on_injected)
    ax1.set_title(title_name)

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
# Full scan mode
# ---------------------------------------------------------------------------

ALL_FNL_VALUES = ['0000', '0001', '0010', '0032', '0100', '1000']


def run_scan(args, freq, bn, ell_range, jackknife, center_on_injected=True):
    results = []

    for gnl in [False, True]:
        estimator_label = 'cubic' if gnl else 'linear'
        for noise in ['with', 'without']:
            data_path = resolve_path('liuto', noise)

            print(f"\n  Computing reference (null) covariance for "
                  f"estimator={estimator_label} noise={noise}...")
            null_sims_bl, null_fiducial_bl = load_experiment_data(
                args.bk, freq, bn, ell_range, args.simn, data_path,
                args.id, jackknife, fnl='0000', calc_gnl=gnl)
            null_amplitudes = null_sims_bl / null_fiducial_bl
            reference_cov = compute_covariance_matrix(null_amplitudes)

            for fnl in ALL_FNL_VALUES:
                injected_value = parse_injected_value(fnl, gnl)
                label = f'estimator={estimator_label} noise={noise} fnl={fnl}'

                try:
                    if fnl == '0000':
                        # Same sims as the reference-covariance set: use
                        # leave-one-out to avoid self-referential leakage.
                        sims_bl, fiducial_bl = null_sims_bl, null_fiducial_bl
                        fnl_hists = run_amplitude_statistics(
                            sims_bl, fiducial_bl, fixed_cov_amplitudes=None)
                    else:
                        sims_bl, fiducial_bl = load_experiment_data(
                            args.bk, freq, bn, ell_range, args.simn, data_path,
                            args.id, jackknife, fnl=fnl, calc_gnl=gnl)
                        fnl_hists = run_amplitude_statistics(
                            sims_bl, fiducial_bl, fixed_cov_amplitudes=reference_cov)

                except FileNotFoundError as e:
                    print(f'  [skip] {label}: {e}')
                    for diag in [True, False]:
                        results.append({
                            'estimator': estimator_label, 'noise': noise, 'injected_fnl': fnl,
                            'diag': diag, 'injected_value': injected_value,
                            'status': 'missing_files',
                        })
                    continue

                for diag in [True, False]:
                    data = np.asarray(fnl_hists[diag])
                    stats = compute_summary_stats(data, injected_value=injected_value)
                    results.append({
                        'estimator': estimator_label, 'noise': noise, 'injected_fnl': fnl,
                        'diag': diag, 'injected_value': injected_value,
                        'n_sims': len(data), 'status': 'ok', **stats,
                    })
                    print(f'  {label} diag={diag}: '
                          f"mu={stats['fit_mu']:.4e}  sigma={stats['fit_sigma']:.4e}  "
                          f"median={stats['median']:.4e}  (n={len(data)})")

                    tag = (f'b3d_1var_nu{freq}_lx0_nobl_lcdm{jackknife}_{ell_range}_b{bn}'
                           f'_{noise}noise{"_gnl" if gnl else ""}_fnl{fnl}')
                    outpath = os.path.join(args.outdir, f'{tag}_diag{diag}.png')
                    title_name = (f'{freq}GHz lCDM ell {ell_range}, {bn} bins, diag={diag}, '
                                  f'noise={noise}, injected fnl={fnl}')
                    plot_fnl_hists(data, injected_value, title_name, outpath,
                                    param_name='gNL' if gnl else 'fNL',
                                    center_on_injected=center_on_injected)

    return pd.DataFrame(results)


def print_scan_table(df):
    ok = df[df['status'] == 'ok'].copy()
    display_cols = ['estimator', 'noise', 'injected_fnl', 'injected_value', 'diag',
                     'fit_mu', 'fit_sigma', 'bias_mu', 'median', 'p16', 'p84',
                     'bias_median', 'n_sims']
    for c in ['injected_value', 'fit_mu', 'fit_sigma', 'bias_mu', 'median', 'p16', 'p84', 'bias_median']:
        ok[c] = ok[c].map(lambda x: f'{x:.4e}' if pd.notnull(x) else '')

    print("\n" + "=" * 140)
    print("SUMMARY STATISTICS (copy-paste table) — bias_* = recovered - injected, i.e. 0 = perfectly unbiased")
    print("=" * 140)
    print(ok[display_cols].to_string(index=False))
    print("=" * 140)

    missing = df[df['status'] == 'missing_files']
    if len(missing):
        print(f"\n{len(missing)} combo(s) skipped due to missing files:")
        for _, row in missing.iterrows():
            print(f"  estimator={row['estimator']} noise={row['noise']} "
                  f"fnl={row['injected_fnl']} diag={row['diag']}")
# ---------------------------------------------------------------------------
# CLI
#
def build_parser():
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
    if args.covdata == 'None':
        args.covdata = None

    if not args.scan_all:
        args.data = resolve_path(args.data, args.noise)
        if args.covdata is not None:
            args.covdata = resolve_path(args.covdata, args.noise)

    if args.outdir is None:
        args.outdir = 'scan_figs' if args.scan_all else f'{"gnl" if args.gnl else "fnl"}_figs_{args.noise}noise'
    os.makedirs(args.outdir, exist_ok=True)

    if args.simn == 0:
        args.simn = None

    return args
def preflight_check(bk_type, freq, bn, ell_range, jackknife, data_path, covdata_path, fnl, calc_gnl, label=''):
    """
    Prints every path/pattern this combo will look for, and reports
    existence, WITHOUT loading any actual file contents. Call this before
    load_experiment_data so missing paths are obvious up front.
    """
    print(f"\n  --- preflight [{label}] ---")
    print(f"  data_path   : {data_path}   exists={os.path.isdir(data_path) if data_path else 'N/A (None)'}")
    print(f"  covdata_path: {covdata_path}   exists={os.path.isdir(covdata_path) if covdata_path else 'N/A (None or not used)'}")

    if bk_type != 'B33y':
        print(f"  [WARN] unrecognized bk_type '{bk_type}', pattern below will not apply")
        return False

    tag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm{jackknife}_{ell_range}_b{bn}'
    if fnl != '':
        sim_pattern = os.path.join(data_path or '<None>', f'{tag}_*_fnl{fnl}.dat')
    else:
        sim_pattern = os.path.join(data_path or '<None>', f'{tag}_*.dat')
    print(f"  sim glob    : {sim_pattern}")
    n_matches = len(glob.glob(sim_pattern)) if data_path and os.path.isdir(data_path) else 0
    print(f"                -> {n_matches} file(s) matched")

    nojacktag = f'b3d_1var_nu{freq}_lx0_nobl_lcdm_{ell_range}_b{bn}'
    fiducial_base = (data_path if covdata_path is None else covdata_path) or '<None>'
    template_base = nojacktag.replace("b3d", "mb3d_fnle")
    if calc_gnl:
        template_base += '_cubic'
    fiducial_path = os.path.join(fiducial_base, f'{template_base}.dat')
    print(f"  fiducial    : {fiducial_path}   exists={os.path.exists(fiducial_path)}")

    all_ok = bool(data_path and os.path.isdir(data_path) and n_matches > 0 and os.path.exists(fiducial_path))
    print(f"  --- preflight result: {'OK' if all_ok else 'MISSING SOMETHING'} ---")
    return all_ok

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