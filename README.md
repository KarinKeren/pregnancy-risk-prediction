# pregnancy-risk-prediction
NLP + Machine learning models for predicting hypertensive complications in pregnancy using structured labs and clinical notes.

📊 Methodology

EDA (Tabular + Text)

simple_eda() summarizes missing data, datatypes, target prevalence.

Missing columns >99% dropped as raw values; kept as is_missing flags.

Clinical notes parsed into 7 canonical sections (Complaints, Risk Factors, Findings, Labs/Imaging, Medications, Vitals, Recommendations).

Feature Engineering:

Unit normalization (UnitNormalizer) for HGB, HCT, WBC, PLT, etc.

Binary abnormal flags (BinaryFlagsAdding) and age >38 risk flag.

Derived ratios: NLR, PLR; BP features (MAP, pulse pressure).

History aggregates: counts in 4m vs 24m, new vs chronic flag.

Text features: TF-IDF on cleaned notes; also keyword/risk-factor flags.

Modeling:

Train/test split with stratification.

Models trained on:

Structured tabular features

Text-derived features

Fused predictions (tab + text)

Evaluation:

Metrics: ROC-AUC, PR-AUC (metrics() in utils).

Budget-constrained recall & PPV (recall_ppv_threshold_at_k()).

Plots: ROC curve, PR curve, Recall–Budget, PPV–Budget.

✅ Results (example)

Tabular only: ROC-AUC ≈ 0.64, PR-AUC ≈ 0.09

Text only (risk/complication section): ROC-AUC ≈ 0.97, PR-AUC ≈ 0.70

Fused (50/50): ROC-AUC ≈ 0.90, PR-AUC ≈ 0.52

Budget: top 20% patients tested → detect ~70% of true cases, ~5× higher yield vs random testing.

📌 Recommendations

Exclude \check more for leaky variables (diagnosis matches, explicit “preeclampsia” mentions in text).

Use risk-factor text features (proteinuria, edema, headache, etc.) with class (report or not in clinical txt) extracted via LLM.

Focus on budget-aware thresholds: ~20% testing achieves best balance between recall and efficiency.

Future: compare multiple model types (XGB, logistic, simple NN), external validation, and more advanced text feature extraction (e.g., LLM-based risk tagging).

👩‍⚕️ Clinical Perspective

This approach shows how combining structured labs with cleaned clinical notes can create an early screening tool at ~week 15, concentrating limited resources on the highest-risk patients.
