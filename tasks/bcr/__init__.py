"""BCR (biochemical-recurrence) survival task — two-stage hierarchical MIL.

Survival modelling of prostate biochemical recurrence with an ``NLLSurvLoss``
(discrete-time, 4 bins) two-stage pipeline:

Run order
---------
1. ``train_stage_1``      — train the flat region-level ``RegionEncoder``.
2. ``infer_stage_1_attn`` — run the best stage-1 checkpoint over every cohort and
                            save the per-region attention weights to disk
                            (``region_attn/<model_type>[...]/<patient>.npy``).
3. ``train_stage_2``      — train the hierarchical (region -> WSI) encoder; the
                            stage-1 attention selects the top-``MAX_REGION_NUM``
                            regions per slide.
4. ``infer_stage_2``      — score the trained stage-2 model, dump per-patient risk
                            CSVs.
"""
