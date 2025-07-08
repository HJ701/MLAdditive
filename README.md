# ML-Driven Additive Selection for Polyolefins

This project develops a machine learning system to recommend optimal additive formulations for polyethylene and polypropylene grades, enhancing their lifetime performance under specific environmental conditions.

## 🔍 Objective

To intelligently predict the **ideal combination and dosage** of additives (e.g., antioxidants, carbon black, waxes) based on resin type and environmental exposure, using ML regression models.

---

## 🧩 Project Components

### 1. Dataset Development
- **Synthetic and empirical data** simulation covering:
  - Base resin: Polyethylene / Polypropylene
  - Additives: Primary & Secondary AO, Carbon Black, Waxes, etc.
  - Environment: Temp, UV, ClO₂, Water, Pressure
  - Output: Predicted Lifetime

### 2. Model Development
- ML regression pipeline:
  - Feature preprocessing & encoding
  - Lifetime prediction using regression models (e.g., XGBoost, MLP, etc.)
  - Evaluation: R², MAE, visual plots

### 3. Degradation Logic (Optional)
- Surrogate models or analytical equations (literature-informed)

### 4. GUI Interface (Optional)
- A simple interface to input resin type & environment, and output optimal additive formulation

---

## 📈 Deliverables

| Item            | Description                                         |
|-----------------|-----------------------------------------------------|
| Dataset         | Clean, engineered additive-performance data         |
| ML Model        | Trained models with performance reports             |
| GUI             | Optional predictive interface (under development)   |
| Final Report    | Documentation, results, codebase (this repo)        |

---

## 🧠 Bonus Extensions
- **Cost-Performance Optimization:** Minimize cost while maximizing lifetime
- **Surrogate Physics Models:** Incorporate chemical degradation logic

---

## 📄 License & Attribution

This project is internal to **Borouge** and protected under institutional research collaboration. Not for public use unless authorized.

