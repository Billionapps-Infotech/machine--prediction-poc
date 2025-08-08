
# Machine Failure Prediction — POC (v3.1)

**New in v3.1**
- Top-3 class **probabilities** column (e.g., `Mechanical Wear (52%), Electrical Fault (30%), No Failure (18%)`)
- **Confusion matrix** card with metrics (accuracy, macro precision/recall/F1)
- **Seasonality control**: sidebar slider `Summer ambient boost (°C)` drives Jul/Aug heat and load in forecasting
- **Row color intensity** based on prediction confidence (red=defect, green=no-failure)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional LLM explanations
Create `.env` with:
```
OPENAI_API_KEY=sk-...
```
(When present, explanations can be made more natural; current build focuses on transparent drivers.)
