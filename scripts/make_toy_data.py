from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.n
    age = rng.normal(72, 12, n).clip(35, 95)
    gender = rng.choice([0, 1], size=n)
    heart_failure = np.ones(n)
    hypertension = rng.binomial(1, sigmoid((age - 65) / 10))
    prior_mi = rng.binomial(1, sigmoid((age - 70) / 14 + 0.5 * gender))
    creatinine = rng.lognormal(np.log(1.2), 0.35, n) + 0.25 * hypertension
    egfr = (95 - 0.65 * age - 12 * (creatinine - 1.0) + rng.normal(0, 8, n)).clip(5, 120)
    bun = (15 + 0.35 * (100 - egfr) + rng.normal(0, 5, n)).clip(3, 120)
    hemoglobin = (13.5 - 0.025 * (age - 60) - 0.018 * (100 - egfr) + rng.normal(0, 1.3, n)).clip(5, 18)
    glucose = rng.normal(130, 35, n).clip(60, 360)
    hba1c = rng.normal(6.3, 1.4, n).clip(4.5, 13.5)
    sodium = rng.normal(138, 4, n)
    potassium = rng.normal(4.2, 0.5, n)
    troponin = rng.lognormal(-2.2, 0.9, n) + 0.08 * prior_mi
    bnp = rng.lognormal(6.1, 0.9, n)
    nyha = rng.choice([1, 2, 3, 4], size=n, p=[0.12, 0.35, 0.38, 0.15])
    lvef = rng.normal(38, 12, n).clip(10, 70)

    diabetes_current = rng.binomial(1, sigmoid((glucose - 135) / 25 + (hba1c - 6.5)))
    renal_current = rng.binomial(1, sigmoid((1.4 - egfr / 60) + 0.9 * diabetes_current + 0.2 * hypertension))
    anemia_current = rng.binomial(1, sigmoid((11.5 - hemoglobin) + 0.5 * renal_current))
    cad_current = rng.binomial(1, sigmoid(-1.2 + 0.5 * prior_mi + 0.4 * diabetes_current + 0.02 * (age - 65)))
    af_current = rng.binomial(1, sigmoid(-1.5 + 0.45 * cad_current + 0.03 * (age - 65) + 0.2 * nyha))
    copd_current = rng.binomial(1, sigmoid(-2.0 + 0.02 * (age - 65) + 0.2 * gender + rng.normal(0, 0.4, n)))

    dialysis = rng.binomial(1, sigmoid(-4 + 2.2 * renal_current + 0.03 * (100 - egfr)))
    crrt = rng.binomial(1, sigmoid(-5 + 2.5 * renal_current + 1.5 * dialysis))
    pci = rng.binomial(1, sigmoid(-4 + 2.0 * cad_current + 0.6 * prior_mi))
    cabg = rng.binomial(1, sigmoid(-5 + 1.5 * cad_current + 0.7 * prior_mi))
    mechanical_ventilation = rng.binomial(1, sigmoid(-4 + 1.2 * copd_current + 0.4 * nyha))
    transfusion = rng.binomial(1, sigmoid(-4 + 1.6 * anemia_current + 0.6 * renal_current))
    diuretic = rng.binomial(1, sigmoid(-0.2 + 0.7 * nyha + 0.4 * renal_current))
    acei_arb_arni = rng.binomial(1, sigmoid(0.2 + 0.2 * (70 - age) / 10 - 0.4 * renal_current))
    beta_blocker = rng.binomial(1, sigmoid(0.1 + 0.6 * cad_current + 0.3 * af_current))
    anticoagulant = rng.binomial(1, sigmoid(-1.5 + 2.0 * af_current))
    insulin = rng.binomial(1, sigmoid(-2.0 + 2.0 * diabetes_current + 0.1 * (glucose - 130) / 30))
    bronchodilator = rng.binomial(1, sigmoid(-2.0 + 2.0 * copd_current))

    num_prior_admissions = rng.poisson(1.1 + 0.5 * nyha + 0.6 * renal_current)
    days_since_last_admission = rng.exponential(180, n).clip(0, 1200)
    los_days = rng.gamma(3 + nyha * 0.7, 2.0, n).clip(1, 60)
    icu_stay = rng.binomial(1, sigmoid(-2 + 0.6 * nyha + 0.6 * renal_current + 0.5 * copd_current))
    delta_t_days = rng.choice([30, 90, 180], size=n, p=[0.25, 0.50, 0.25])

    def future_from(current, risk):
        return rng.binomial(1, np.maximum(current * 0.78, risk))

    renal_future = future_from(renal_current, sigmoid(-2.0 + 0.03 * (100 - egfr) + 0.8 * diabetes_current + 0.4 * diuretic))
    anemia_future = future_from(anemia_current, sigmoid(-2.2 + 0.5 * renal_current + 0.05 * (12 - hemoglobin)))
    diabetes_future = future_from(diabetes_current, sigmoid(-2.3 + 0.035 * (glucose - 120) + 0.9 * (hba1c > 6.5)))
    cad_future = future_from(cad_current, sigmoid(-2.0 + 0.6 * diabetes_current + 0.5 * prior_mi + 0.5 * troponin))
    af_future = future_from(af_current, sigmoid(-2.1 + 0.5 * cad_current + 0.3 * nyha + 0.015 * (age - 65)))
    copd_future = future_from(copd_current, sigmoid(-2.2 + 0.4 * mechanical_ventilation + 0.3 * bronchodilator))

    df = pd.DataFrame({
        "subject_id": [f"S{i//2:05d}" for i in range(n)],
        "age": age, "gender": gender, "heart_failure": heart_failure, "hypertension": hypertension, "prior_mi": prior_mi,
        "creatinine": creatinine, "egfr": egfr, "bun": bun, "hemoglobin": hemoglobin, "glucose": glucose, "hba1c": hba1c,
        "sodium": sodium, "potassium": potassium, "troponin": troponin, "bnp": bnp, "nyha": nyha, "lvef": lvef,
        "af_current": af_current, "renal_current": renal_current, "copd_current": copd_current, "diabetes_current": diabetes_current, "cad_current": cad_current, "anemia_current": anemia_current,
        "diuretic": diuretic, "acei_arb_arni": acei_arb_arni, "beta_blocker": beta_blocker, "anticoagulant": anticoagulant, "insulin": insulin, "bronchodilator": bronchodilator,
        "dialysis": dialysis, "crrt": crrt, "pci": pci, "cabg": cabg, "mechanical_ventilation": mechanical_ventilation, "transfusion": transfusion,
        "num_prior_admissions": num_prior_admissions, "days_since_last_admission": days_since_last_admission, "los_days": los_days, "icu_stay": icu_stay,
        "delta_t_days": delta_t_days,
        "af_future": af_future, "renal_future": renal_future, "copd_future": copd_future, "diabetes_future": diabetes_future, "cad_future": cad_future, "anemia_future": anemia_future,
    })
    # Feature missingness: labs not always observed.
    for col in ["hba1c", "troponin", "bnp", "lvef", "bun"]:
        miss = rng.random(n) < 0.35
        df.loc[miss, col] = np.nan
    # Simulate hospital partial labels: cad/anemia future labels unobserved for a subset.
    partial = rng.random(n) < 0.45
    df.loc[partial, "cad_future"] = np.nan
    df.loc[partial, "anemia_future"] = np.nan
    # Simulate no medication/procedure for some rows, like structural view gaps.
    med_proc_cols = ["diuretic", "acei_arb_arni", "beta_blocker", "anticoagulant", "insulin", "bronchodilator", "dialysis", "crrt", "pci", "cabg", "mechanical_ventilation", "transfusion"]
    structural = rng.random(n) < 0.25
    df.loc[structural, med_proc_cols] = np.nan
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved toy data to {args.output} with shape {df.shape}")


if __name__ == "__main__":
    main()
