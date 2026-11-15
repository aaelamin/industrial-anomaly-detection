# Saved-map arithmetic check

An additional CPU-only validity diagnostic checked all 1,152 committed AeBAD maps: three seeds, initialization and epoch 500, with 32 calibration and 160 development maps per endpoint. No source images, masks or models were loaded. The protocol and a 10-minute cap were frozen before reading the arrays; execution took 11.93 seconds.

The check compared raw float32 values, float32 addition of one, and float64 addition of one. No nonconstant map became constant. The largest per-map loss of distinct values after float32 addition was 0.6523%, below the frozen 1% follow-up flag. The largest absolute arithmetic difference was 2.98023224e-8. Across trained maps, 288 nonzero pixels rounded to exactly one after the shift; four raw pixels were already zero. These counts concern the stored 256-by-256 maps, not native-resolution pixels or annotated defects.

This is a limited arithmetic audit. It does not independently recompute native interpolation, thresholded region outcomes, or pixel metrics; it does not prove that every reported metric is numerically unchanged. The previously reported histogram bounds apply to the evaluator's transformed scores. The audit did not trigger its predeclared further-evaluation criterion. Existing results and the frozen evaluator remain unchanged.
