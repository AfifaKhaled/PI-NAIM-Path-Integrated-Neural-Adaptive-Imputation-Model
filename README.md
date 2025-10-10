# PI-NAIM (Path-Integrated Neural Adaptive Imputation Model)

This is a **patched implementation** of the PI-NAIM architecture described in the paper:

> PI-NAIM: Path-Integrated Neural Adaptive Imputation Model  
> A dual-path framework combining statistical (MICE) and neural (GAIN + temporal analysis) methods for robust missing data imputation.

---

## 🔑 Features
- **Dynamic path routing**  
  Routes samples to **MICE** (low missingness) or **GAIN** (high/complex missingness).
- **Temporal GAIN with WGAN-GP**  
  Supports adversarial training and temporal self-attention.
- **Cross-path attention fusion**  
  Combines embeddings from both paths.
- **Task-supervised adaptive fusion**  
  Balances imputation quality with downstream tasks.
- **Curriculum masking**  
  Training schedule progresses MCAR → MAR → MNAR.
- **Uncertainty quantification**  
  Via MC-dropout in GAIN and bootstrap variance in MICE.
- **Multi-task objective**  
  Joint loss with homoscedastic uncertainty weighting.

---

## 📂 Repository Structure
