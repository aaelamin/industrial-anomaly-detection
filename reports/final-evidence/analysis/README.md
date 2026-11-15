# Regenerate the result plots

From the kit root, run `python analysis/plot_tables.py --tables team-tables --output regenerated-figures`. Use the pinned NumPy and Matplotlib environment described in the reproduction setup. Choose a new output folder; existing figure results are preserved. This script verifies the table hashes and recomputes every plotted mean and SD from the seed rows. It loads no model, image or mask.

The complete model rerun is a separate command in `reproduction/README.md`; the plotting command uses the supplied verified result tables. The delivered SVGs are also editable directly.

Regenerate the readable AeBAD tables with `python analysis/export_transfer_tables.py --bundle transfer-evidence --output regenerated-transfer-tables`. This command uses only the Python standard library and saved numeric evidence. Choose a new output directory.
