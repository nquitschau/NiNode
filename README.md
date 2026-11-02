# NiNode

**Node-level (parcel-wise) functional connectivity metrics from XCP-D outputs.**

NiNode takes the parcellated outputs of [XCP-D](https://xcp-d.readthedocs.io) (Pearson correlation matrices and mean BOLD time series) and computes one value per parcel for each metric. Outputs are saved as BIDS-style derivatives next to the XCP-D files, ready for group statistics or spatial colocalization analyses.

> **Status:** early version (v0.1.0). Interface and outputs may change in future releases.

## Metrics

| Metric | Input | Description |
|---|---|---|
| Binary degree centrality (`degree`) | connectivity matrix | Number of positive connections above a correlation threshold (default r > 0.25) per parcel |
| Positive weighted global connectivity (`pwgc`) | connectivity matrix | Mean of positive Fisher-z transformed correlations per parcel, back-transformed to r |
| Temporal variability (`tv`) | mean time series | 1 − mean similarity of a parcel's FC profile across non-overlapping 60 s windows (window length adjusted to the TR) |

Only positive correlations are used for degree and pWGC, since global signal regression can introduce negative correlations (Murphy et al., 2009).

## Installation

```bash
git clone https://github.com/nquitschau/ninode.git
cd ninode
pip install -r requirements.txt
```

## Usage

```bash
python NiNode.py \
    -i /path/to/xcpd_derivatives \
    -b /path/to/bids_dataset \
    -o /path/to/output \
    -a TianSchaefer40017NS2 \
    -t 0.25
```

| Argument | Description |
|---|---|
| `-i`, `--input` | Root directory of the XCP-D derivatives (required) |
| `-b`, `--bids` | Original BIDS dataset, used to read the TR from `*_bold.json` (required) |
| `-o`, `--output` | Output root directory (default: same as input) |
| `-a`, `--atlas` | Atlas name as used in the XCP-D filenames (default: `TianSchaefer40017NS2`) |
| `-t`, `--threshold` | Positive correlation threshold for degree centrality (default: 0.25) |
| `--overwrite` | Recompute and overwrite existing outputs |

### Expected input

```
xcpd_derivatives/
└── sub-<label>/ses-<label>/func/
    ├── *_atlas-<atlas>_stat-pearsoncorrelation_relmat.tsv
    └── *_atlas-<atlas>_*mean_timeseries.tsv
```

If a concatenated file (without `run-`) exists for a subject/session/task, it is used; otherwise the first available run is used.

### Output

One TSV per subject, session, task and metric, in the same folder structure as the input:

```
*_atlas-<atlas>_stat-degree_bold.tsv
*_atlas-<atlas>_stat-pwgc_bold.tsv
*_atlas-<atlas>_*tv_bold.tsv
```

Each file has two rows: `Node` (parcel labels) and `Value` (metric per parcel). All outputs pass a QC check (completeness, no NaNs, valid value range).

## Roadmap

- Choose metrics via `--metrics`
- Configurable TV window length
- Additional node-level metrics
- pip-installable package with a `ninode` command

## References

- Cole, M. W., Pathak, S., & Schneider, W. (2010). Identifying the brain's most globally connected regions. *NeuroImage*, 49(4), 3132–3148.
- Cole, M. W., & Schneider, W. (2007). The cognitive control network: Integrated cortical regions with dissociable functions. *NeuroImage*, 37, 343–360.
- Cox, R. W. (1996). AFNI: Software for analysis and visualization of functional magnetic resonance neuroimages. *Computers and Biomedical Research*, 29(3), 162–173.
- Craddock, R. C., & Clark, D. J. (2016). Optimized implementations of voxel-wise degree centrality and local functional connectivity density mapping in AFNI. *GigaScience*, 5(Suppl 1).
- Hutchison, R. M., et al. (2013). Dynamic functional connectivity: Promise, issues, and interpretations. *NeuroImage*, 80, 360–378.
- Murphy, K., Birn, R. M., Handwerker, D. A., Jones, T. B., & Bandettini, P. A. (2009). The impact of global signal regression on resting state correlations: Are anti-correlated networks introduced? *NeuroImage*, 44(3), 893–905.
- Zou, H., & Yang, J. (2021). Temporal variability-based functional brain lateralization study in ADHD. *Journal of Attention Disorders*, 25(6), 839–847.