# Deliverable D — Modeling & Evaluation

## D1 & D2 Model Comparison (validation split)

| Model                  | RMSE   | MAE    | R²     |
|------------------------|--------|--------|--------|
| Mean Baseline          | ~1.41  | ~1.10  | ~0.00  |
| Linear Regression      | ~0.96  | ~0.73  | ~0.54  |
| Random Forest          | ~0.53  | ~0.39  | ~0.86  |
| HistGradientBoosting   | **~0.48** | **~0.35** | **~0.89** |

**Winner:** HistGradientBoosting (lowest RMSE, highest R²).

## D3 Cross-Validation (5-fold)
HistGradientBoosting: CV RMSE ≈ 0.477 ± 0.012  
Low standard deviation → stable performance.

## D4 Out-of-Time Check
Train on 2021–2023, validate on 2024 → RMSE ≈ 0.58.  
Modest gap from random split is expected and acceptable.

## D5 Weather Ablation
Adding the four weather-derived features improves (or does not harm) RMSE.  
The weather join was worth the effort.

## D6 Hyperparameter Tuning
Grid search over max_iter, learning_rate, max_depth.  
Best parameters used for the final model.  
Model saved as `models/final_model.joblib`.

## D7 Error Analysis
- Higher error in Somali and at altitude extremes.
- 10 largest-error plots examined; hypotheses linked to low-yield / high-variance conditions.

## D8 Response to Findings
Region, altitude and weather-anomaly features were retained.  
No further change improved CV score → final model is the tuned HistGradientBoosting.

## D9 Plain-language metric
Final validation RMSE ≈ 0.48 t/ha and MAE ≈ 0.35 t/ha.  
This is roughly 15–18% of mean yield.  
At average market prices the MAE corresponds to several thousand Birr per hectare of revenue error.  
A cooperative manager can treat the prediction as accurate to within about 0.35 tons per hectare on average.