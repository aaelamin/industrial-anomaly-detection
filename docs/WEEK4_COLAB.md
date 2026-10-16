# Running Week 4 on Colab

This is the original setup or planning record. See the [completed results](../reports/README.md) for the later experiments and conclusions.

Open `notebooks/week4_study.ipynb` using Colab's Upload notebook option. Select a GPU runtime. The upload package is `work/colab-study/week4_study.zip` on the computer where it was built; it is not committed to GitHub because it includes the bottle images.

Keep at least 25 GB free in Google Drive. Use only one active notebook for this study folder.

1. Run notebook steps 1–4. Connect Drive, upload the package, install the environment, and prepare the short recovery test.
2. Disconnect and delete the runtime. Reconnect with the same GPU type and repeat steps 1–3. The package is loaded from Drive this time.
3. Skip step 4 and run step 5. It must report that the resumed epoch matches exactly.
4. Run step 6 to download and save the full dataset. In step 7, set `START_STUDY = True` and run the cell.
5. After later interruptions, reconnect and run steps 1, 2, 3, 6, and 7. The runner skips completed runs and restores the unfinished one.

The notebook cannot reconnect itself or guarantee GPU availability. The T4 bottle benchmark estimated about 70 hours of active training for all 45 runs, excluding most setup and evaluation time.

## Saved files

Drive folder: `MyDrive/CISC473/week4_wrn50`.

Each `runs/<category>/seed_<number>` folder contains the final metrics, per-image scores, anomaly maps, selected heatmaps, training history, and final trainable weights. A checked `complete.json` marks a finished run. Scores use the maximum value in each anomaly map, as in Week 3.

Checkpoints include trainable parameters, Adam state, epoch history, and the Python, NumPy, PyTorch, CUDA, and sampling-generator states. They are saved every 25 epochs. The previous checkpoint stays available until the new copy is verified. An interrupted write can fall back to the earlier readable checkpoint. Up to 25 epochs of work may need repeating.

The frozen backbone is reconstructed from the checksum-verified pretrained weights. Final `model.pt` files contain `trainable_state` and the configuration, rather than another copy of the backbone. Rebuild with `week3.build_model(config, 'cuda')`, then load with `scripts.week4_colab.restore_model(model, saved['trainable_state'])`.

Completed runs keep their final weights and outputs; their optimizer checkpoints are removed after the result copies pass checksum checks. Cached features stay on runtime disk. Settings, source hashes, and package versions are recorded in `study.json`; changing these blocks resume to prevent mixing different experiments.

The recovery test is separate from the 500-epoch runs. It compares a freshly restored second epoch with an uninterrupted second epoch, including the loss, model, optimizer, and random states. The notebook also checks that the restore is performed in a different runtime. It needs the same GPU type for this exact comparison. A passing local test does not establish that Drive recovery works; step 5 must pass in Colab before training.
