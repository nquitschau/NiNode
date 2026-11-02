#!/usr/bin/env python3

"""
This script computes parcel-wise global metrics from XCP-D Pearson correlation matrices.

Metrics
-------
1) degree centrality: number of positive suprathreshold connections per parcel
2) positive weighted global connectivity: mean Fisher-z transformed connectivity per parcel, back-transformed to Pearson r
3) temporal variability: variability of parcel-wise functional connectivity profiles across non-overlapping 60s windows (adjusted for TR)
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# Command line arguments
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute parcel-wise Global Connectivity metrics from XCP-D Pearson correlation matrices."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Root directory containing XCP-D derivatives."
    )

    parser.add_argument(
        "-b",
        "--bids",
        type=Path,
        required=True,
        help="Path to the original BIDS source dataset (necessary for TR extraction)."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Root directory for the output derivatives. Defaults to input directory.",
    )

    parser.add_argument(
        "-a",
        "--atlas",
        default="TianSchaefer40017NS2",
        help="Atlas name used in the input filenames (default: TianSchaefer40017NS2)."
    )

    parser.add_argument(
        "-t",
        "--threshold",
        default=0.25,
        type=float,
        help="Positive Pearson correlation threshold (default: 0.25)."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metric files."
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input directory does not exist:\n{args.input}")

    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0 and 1.")

    if args.output is None:
        args.output = args.input

    args.output.mkdir(parents=True, exist_ok=True)

    return args


# =============================================================================
# File handling
# =============================================================================

def get_input_pattern(atlas):
    return (
        f"sub-*/ses-*/func/"
        f"*atlas-{atlas}_stat-pearsoncorrelation_relmat.tsv"
    )

def get_timeseries_pattern(atlas):
    return (
        f"sub-*/ses-*/func/"
        f"*atlas-{atlas}*mean_timeseries.tsv"
    )


def select_connectivity_matrices(files):
    # If a combined matrix exists (without run-), it will be selected.
    # Otherwise the first available run is used.

    print("\nSelecting connectivity matrices...")

    selected = {}

    for file in files:
        entities = {}

        for part in file.stem.split("_"):
            if "-" in part:
                key, value = part.split("-", 1)
                entities[key] = value

        key = (
            entities.get("sub"),
            entities.get("ses"),
            entities.get("task"),
            entities.get("atlas"),
        )

        combined = "run-" not in file.name

        if combined:
            selected[key] = file

        elif key not in selected:
            selected[key] = file

    return sorted(selected.values())


def select_timeseries(files):
    # Same selection logic as for connectivity matrices.

    print("\nSelecting mean time series...")

    selected = {}

    for file in sorted(files):
        entities = {}

        for part in file.stem.split("_"):
            if "-" in part:
                key, value = part.split("-", 1)
                entities[key] = value

        key = (
            entities.get("sub"),
            entities.get("ses"),
            entities.get("task"),
            entities.get("atlas"),
        )

        combined = "run-" not in file.name

        if combined:
            selected[key] = file

        elif key not in selected:
            selected[key] = file

    return sorted(selected.values())

def get_subject(file):
    for part in file.parts:
        if part.startswith("sub-"):
            return part

    raise ValueError(f"Could not determine subject from {file}")

def get_tr(subject, bids_dir):
    json_files = list(
        (bids_dir / subject).rglob("*bold.json")
    )

    trs = set()

    for json_file in json_files:
        with open(json_file) as f:
            metadata = json.load(f)

        if "RepetitionTime" in metadata:
            trs.add(float(metadata["RepetitionTime"]))

    if len(trs) == 0:
        raise ValueError(f"No TR found for {subject}")

    if len(trs) > 1:
        raise ValueError(
            f"Multiple TRs found for {subject}: {sorted(trs)}"
        )

    return trs.pop()


# =============================================================================
# Connectivity matrix
# =============================================================================

def load_connectivity_matrix(input_file: Path) -> pd.DataFrame:

    print("\nLoading and checking parcel-wise connectivity matrix:")
    print(f"  {input_file.name}")

    matrix = pd.read_csv(input_file, sep="\t", index_col=0)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Connectivity matrix is not square: {input_file} "
            f"({matrix.shape[0]} × {matrix.shape[1]})"
        )

    if list(matrix.index.astype(str)) != list(matrix.columns.astype(str)):
        raise ValueError(
            f"Row and column labels do not match: {input_file}"
        )

    values = matrix.to_numpy(dtype=float)

    if np.isnan(values).any():
        raise ValueError(
            f"Connectivity matrix contains NaN values: {input_file}"
        )

    if not np.allclose(values, values.T, equal_nan=True, atol=1e-6):
        raise ValueError(
            f"Connectivity matrix is not symmetric: {input_file}"
        )

    return matrix


# =============================================================================
# Matrix preparation
# =============================================================================

def remove_diagonal(matrix: pd.DataFrame) -> np.ndarray:

    print("Removing diagonal")

    values = matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, np.nan)

    return values


# =============================================================================
# Binary Degree Centrality
# =============================================================================

def compute_degree(values: np.ndarray, threshold: float) -> np.ndarray:

    print("\nComputing Binary Degree Centrality (bDC)")
    print("----------------------------------------")
    print("Interpretation:")
    print("  Number of relevant (positive + suprathreshold) connections per parcel.")
    print()
    print("Implementation:")
    print(f"  Binary count of Pearson correlations exceeding a threshold of r > {threshold}.")
    print()
    print("Reference:")
    print("  Cox, R. W. (1996). AFNI: software for analysis and visualization of functional magnetic resonance neuroimages. Computers and Biomedical Research, 29(3), 162-173.")
    print("  Craddock, R. C., & Clark, D. J. (2016). Optimized implementations of voxel-wise degree centrality and local functional connectivity density mapping in AFNI. GigaScience, 5(Suppl 1), s13742-016-0147-0-d.")
    print("  Cole, M. W., Pathak, S., & Schneider, W. (2010). Identifying the brain’s most globally connected regions. NeuroImage, 49(4), 3132–3148.")
    print("----------------------------------------")

    mask = values > threshold

    return np.sum(mask, axis=1).astype(int)


# =============================================================================
# Positive Weighted Global Connectivity
# =============================================================================

def compute_pwgc(values: np.ndarray) -> np.ndarray:

    print("\nComputing Positive Weighted Global Connectivity (pWGC)")
    print("----------------------------------------")
    print("Interpretation:")
    print("  Functional connectivity strength per parcel.")
    print()
    print("Implementation:")
    print("  Mean of positive Pearson correlations after Fisher-z transformation. Only positive connections are considered, since GSR used in XCP-D can inflate negative correlations (Murphy et al., 2009).")
    print()
    print("Reference:")
    print("  Cole, M. W., Pathak, S., & Schneider, W. (2010). Identifying the brain’s most globally connected regions. NeuroImage, 49(4), 3132–3148.")
    print("  Cole, M.W., Schneider, W., 2007. The cognitive control network: integrated cortical regions with dissociable functions. NeuroImage 37, 343–360.")
    print("  Murphy, K., Birn, R. M., Handwerker, D. A., Jones, T. B., & Bandettini, P. A. (2009). The impact of global signal regression on resting state correlations: are anti-correlated networks introduced?. NeuroImage, 44(3), 893–905.")
    print("----------------------------------------")

    pMask = values > 0

    clipped = np.clip(values, -0.999999, 0.999999)
    fisher = np.arctanh(clipped)
    fisher[~pMask] = np.nan

    pwgc = np.zeros(values.shape[0])

    valid = np.any(pMask, axis=1)

    pwgc[valid] = np.tanh(
        np.nanmean(
            fisher[valid],
            axis=1,
        )
    )

    return pwgc


# =============================================================================
# Temporal Variability
# =============================================================================

def profile_correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return np.corrcoef(x, y)[0, 1]

def compute_tv_for_window_length(ts, window_length):

    n_tp, n_parcels = ts.shape
    n_windows = n_tp // window_length

    if n_windows < 2:
        return None

    # Only complete non-overlapping windows are used.
    ts = ts[:n_windows * window_length]

    fc_matrices = []

    for w in range(n_windows):
        start = w * window_length
        stop = start + window_length

        window = ts[start:stop, :]

        fc = np.corrcoef(
            window,
            rowvar=False
        )

        fc_matrices.append(fc)

    fc_matrices = np.asarray(fc_matrices)

    tv = np.full(n_parcels, np.nan)

    for k in range(n_parcels):

        similarities = []

        # Exclude self-correlation of parcel k.
        mask = np.arange(n_parcels) != k

        for i in range(n_windows):
            profile_i = fc_matrices[i, k, mask]

            for j in range(i + 1, n_windows):
                profile_j = fc_matrices[j, k, mask]

                r = profile_correlation(
                    profile_i,
                    profile_j
                )

                if np.isfinite(r):
                    similarities.append(r)

        if similarities:
            tv[k] = 1 - np.mean(similarities)

    return tv

def compute_tv(ts, tr):

    print("\nComputing Temporal Variability (TV)")
    print("----------------------------------------")
    print("Interpretation:")
    print("  Variability of regional functional connectivity patterns over time.")
    print("  Higher TV indicates less stable connectivity patterns across time.")
    print()
    print("Implementation:")
    print("  Parcel-wise BOLD time series are divided into non-overlapping time windows of 60 seconds, with the number of volumes adjusted according to the participant-specific TR.")
    print("  Within each window, functional connectivity is calculated using Pearson correlations.")
    print("  For each parcel, its connectivity profiles are correlated across all pairs of windows. TV is defined as 1 minus the mean correlation between these connectivity profiles.")
    print()
    print("Reference:")
    print("  Zou, H., & Yang, J. (2021). Temporal Variability-Based Functional Brain Lateralization Study in ADHD. Journal of Attention Disorders, 25(6), 839–847.")
    print("  Hutchison, R. M. et al. (2013). Dynamic functional connectivity: Promise, issues, and interpretations. NeuroImage, 80, 360–378.")
    print("----------------------------------------")

    # only change the following setting, if you wanna change the window duration
    window_seconds = 60

    window_length = int(round(window_seconds / tr))
    n_windows = ts.shape[0] // window_length

    print(f"TR         : {tr} s | "
          f"timepoints : {ts.shape[0]} | "
          f"window     : {window_length} volumes | "
          f"windows    : {n_windows}")

    tv = compute_tv_for_window_length(ts, window_length)

    if tv is None:
        raise ValueError(
            f"Not enough timepoints for TV calculation "
            f"(TR={tr}s, window={window_length} volumes)."
        )

    return tv


# =============================================================================
# Define Output
# =============================================================================

def get_output_path(input_file: Path, input_root: Path, output_root: Path, metric: str) -> Path:

    print("Creating BIDS-compatible output file.")

    old = "stat-pearsoncorrelation_relmat.tsv"
    new = f"stat-{metric}_bold.tsv"

    if old not in input_file.name:
        raise ValueError(f"Unexpected input filename: {input_file.name}")

    relative = input_file.relative_to(input_root)

    output_dir = output_root / relative.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input_file.name.replace(old, new)

    return output_dir / output_name


def get_tv_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:

    print("Creating BIDS-compatible TV output file.")

    relative = input_file.relative_to(input_root)

    output_dir = output_root / relative.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input_file.name

    if "mean_timeseries.tsv" not in output_name:
        raise ValueError(f"Unexpected time series filename: {input_file.name}")

    output_name = output_name.replace(
        "mean_timeseries.tsv",
        "tv_bold.tsv"
    )

    return output_dir / output_name

def save_metric(values: np.ndarray, parcel_names: pd.Index, output_file: Path) -> None:

    output = pd.DataFrame(
        [
            parcel_names,
            values,
        ],
        index=["Node", "Value"],
    )

    output.to_csv(output_file, sep="\t", header=False)

    print(f"Saved: {output_file.name}")


# =============================================================================
# Output QC
# =============================================================================

def check_pwgc_output(output_file: Path, n_parcels: int) -> None:

    print(f"Checking pwGC output: {output_file.name}")

    output = pd.read_csv(output_file,
                         sep="\t",
                         header=None,
                         index_col=0)

    if list(output.index) != ["Node", "Value"]:
        raise ValueError(f"Unexpected output structure: {output_file}")

    if output.shape[1] != n_parcels:
        raise ValueError(f"Unexpected number of parcels: {output_file}")

    values = pd.to_numeric(output.loc["Value"],
                           errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"pwGC contains NaN or infinite values: {output_file}")

    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"pwGC values outside expected range [0, 1]: {output_file}")

    print(f"pwGC QC passed ({output_file.name} is readable, complete and within valid bounds).")


def check_degree_output(output_file: Path, n_parcels: int) -> None:

    print(f"Checking Degree output: {output_file.name}")

    output = pd.read_csv(output_file,
                         sep="\t",
                         header=None,
                         index_col=0)

    if list(output.index) != ["Node", "Value"]:
        raise ValueError(f"Unexpected output structure: {output_file}")

    if output.shape[1] != n_parcels:
        raise ValueError(f"Unexpected number of parcels: {output_file}")

    values = pd.to_numeric(output.loc["Value"],
                           errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"Degree contains NaN or infinite values: {output_file}")

    if np.any((values < 0) | (values > n_parcels - 1)):
        raise ValueError(f"Degree values outside expected range [0, {n_parcels - 1}]: {output_file}")

    if not np.allclose(values, np.round(values)):
        raise ValueError(f"Degree contains non-integer values: {output_file}")

    print(f"Degree QC passed ({output_file.name} is readable, complete and within valid bounds).")


def check_tv_output(output_file: Path, n_parcels: int) -> None:

    print(f"Checking TV output: {output_file.name}")

    output = pd.read_csv(output_file,
                         sep="\t",
                         header=None,
                         index_col=0)

    if list(output.index) != ["Node", "Value"]:
        raise ValueError(f"Unexpected output structure: {output_file}")

    if output.shape[1] != n_parcels:
        raise ValueError(f"Unexpected number of parcels: {output_file}")

    values = pd.to_numeric(output.loc["Value"],
                           errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"TV contains NaN or infinite values: {output_file}")

    print(f"TV QC passed ({output_file.name} is readable and complete).")


# =============================================================================
# Processing
# =============================================================================

def process_degree(input_file: Path, input_root: Path, output_root: Path, threshold: float, overwrite: bool) -> bool:

    degree_file = get_output_path(input_file,
                                  input_root,
                                  output_root,
                                  "degree")

    if degree_file.exists() and not overwrite:
        print(f"\nSkipping existing Degree output:"
              f"\n  {degree_file.name}")
        return False

    matrix = load_connectivity_matrix(input_file)
    values = remove_diagonal(matrix)

    degree = compute_degree(values, threshold)

    save_metric(degree,
                matrix.columns,
                degree_file)

    check_degree_output(degree_file, matrix.shape[0])

    return True


def process_pwgc(input_file: Path, input_root: Path, output_root: Path, overwrite: bool) -> bool:

    pwgc_file = get_output_path(input_file,
                                input_root,
                                output_root,
                                "pwgc")

    if pwgc_file.exists() and not overwrite:
        print(f"\nSkipping existing pwGC output:"
              f"\n  {pwgc_file.name}")
        return False

    matrix = load_connectivity_matrix(input_file)
    values = remove_diagonal(matrix)

    pwgc = compute_pwgc(values)

    save_metric(pwgc,
                matrix.columns,
                pwgc_file)

    check_pwgc_output(pwgc_file, matrix.shape[0])

    return True


def process_timeseries(input_file: Path, input_root: Path, output_root: Path, bids_dir: Path, overwrite: bool) -> bool:

    tv_file = get_tv_output_path(input_file,
                                 input_root,
                                 output_root)

    if tv_file.exists() and not overwrite:
        print(f"\nSkipping existing TV output:"
              f"\n  {tv_file.name}")
        
        return False

    subject = get_subject(input_file)

    tr = get_tr(subject, bids_dir)

    print("\nLoading parcel-wise mean time series:")
    print(f"  {input_file.name}")

    timeseries = pd.read_csv(input_file, sep="\t")
    ts = timeseries.to_numpy(dtype=float)

    if ts.ndim != 2:
        raise ValueError(f"Unexpected time-series dimensions: {input_file}")

    if ts.shape[1] == 0:
        raise ValueError(f"No parcels found in time series: {input_file}")

    if not np.isfinite(ts).all():
        raise ValueError(f"Time series contains NaN or infinite values: {input_file}")

    tv = compute_tv(ts, tr)

    save_metric(tv, timeseries.columns, tv_file)

    check_tv_output(tv_file, ts.shape[1])

    return True


# =============================================================================
# Main + Report
# =============================================================================

def main():

    args = parse_arguments()

    start_time = time.time()

    # connectivity matrices
    input_files = sorted(
        args.input.glob(get_input_pattern(args.atlas))
    )

    input_files = select_connectivity_matrices(input_files)

    if not input_files:
        raise FileNotFoundError("No connectivity matrices found.")

    # mean time series
    timeseries_files = sorted(
        args.input.glob(get_timeseries_pattern(args.atlas))
    )

    timeseries_files = select_timeseries(timeseries_files)

    if not timeseries_files:
        raise FileNotFoundError("No mean time series found.")

    # overview
    print("\n========================================")
    print("Parcel-wise Global Connectivity Metrics")
    print("========================================")
    print()
    print(f"Found {len(input_files)} connectivity matrices.")
    print(f"Found {len(timeseries_files)} mean time series.")
    print(f"Atlas     : {args.atlas}")
    print(f"Threshold : r > {args.threshold}")
    print(f"TV window : 60 s")
    print()

    degree_count = 0
    pwgc_count = 0
    tv_count = 0

    degree_failed = 0
    pwgc_failed = 0
    tv_failed = 0

    # bDC
    for number, input_file in enumerate(input_files, start=1):

        print(f"\n[{number}/{len(input_files)}]")

        try:
            degree_created = process_degree(
                input_file=input_file,
                input_root=args.input,
                output_root=args.output,
                threshold=args.threshold,
                overwrite=args.overwrite,   
            )

            degree_count += int(degree_created)

        except Exception as error:
            degree_failed += 1
            print(f"\nERROR\n{error}")

    # pwGC
    for number, input_file in enumerate(input_files, start=1):
    
        print(f"\n[{number}/{len(input_files)}]")
    
        try:
            pwgc_created = process_pwgc(
                input_file=input_file,
                input_root=args.input,
                output_root=args.output,
                overwrite=args.overwrite,   
            )
    
            pwgc_count += int(pwgc_created)
    
        except Exception as error:
            pwgc_failed += 1
            print(f"\nERROR\n{error}")

    # TV
    for number, timeseries_file in enumerate(timeseries_files, start=1):

        print(f"\n[{number}/{len(timeseries_files)}]")

        try:
            tv_created = process_timeseries(
                input_file=timeseries_file,
                input_root=args.input,
                output_root=args.output,
                bids_dir=args.bids,
                overwrite=args.overwrite,
            )

            tv_count += int(tv_created)

        except Exception as error:
            tv_failed += 1
            print(f"\nERROR\n{error}")

    # final reporting
    elapsed = (time.time() - start_time)

    print("\n========================================")
    print("Finished")
    print()
    print(f"Input matrices    : {len(input_files)}")
    print(f"Input time series : {len(timeseries_files)}")
    print(f"Degree files      : {degree_count}")
    print(f"pwGC files        : {pwgc_count}")
    print(f"TV files          : {tv_count}")
    print(f"Failed            : {degree_failed + pwgc_failed + tv_failed}")
    print(f"Elapsed time      : {elapsed/60:.1f} min")
    print()
    print("========================================")


if __name__ == "__main__":
    main()