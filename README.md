
# Machine Failure Prediction — POC (v3.2)

- **Diverse sample data** generation with seasonality & per-defect priors (no single-class dominance).
- Optional **Diversity booster** to flip a few low-confidence rows to their 2nd-best class if a month is >60% one label (for demo variety).
- All v3.1 features retained: top-3 probabilities, confusion matrix, confidence-based row colors, preview bound to month selector.

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
