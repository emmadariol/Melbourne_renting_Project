#!/usr/bin/env python3
"""
Confronta gli ultimi due modelli salvati (pre- e post-retrain) e
stampa quanto sono diversi in termini di PSI tra i dataset di training.
"""

from drift import compare_latest_models
import pandas as pd

if __name__ == "__main__":
    try:
        res = compare_latest_models()
    except Exception as e:
        print(f"❌ Impossibile confrontare modelli: {e}")
        raise SystemExit(1)

    prev_meta = res["previous_model"]
    curr_meta = res["current_model"]
    drift = res["drift"]
    batch_drift = res.get("batch_drift")

    print("=" * 72)
    print("CONFRONTO MODELLI")
    print("=" * 72)
    print(f"Precedente: {prev_meta.get('filename')}  (note: {prev_meta.get('note','')})")
    print(f"Attuale:    {curr_meta.get('filename')}  (note: {curr_meta.get('note','')})")
    print(f"Righe ref prev: {res['reference_rows']} | Righe ref attuale: {res['current_rows']} | Nuove righe: {res['batch_rows']}")

    psi_series = pd.Series(drift["psi"]).sort_values(ascending=False)
    top5 = psi_series.head(5)

    print("\nPSI per feature (top 5):")
    for feat, val in top5.items():
        print(f"  {feat:15s}: {val:.4f}")

    print(f"\nDrift Ratio: {drift['drift_ratio']:.1%}")
    print(f"Features in drift: {drift['drifted_features']}")
    print(f"should_retrain (secondo PSI): {drift['should_retrain']}")

    if batch_drift is not None:
        psi_series_b = pd.Series(batch_drift["psi"]).sort_values(ascending=False)
        top5b = psi_series_b.head(5)

        print("\n--- Drift solo sulle nuove righe (delta) ---")
        print(f"Nuove righe considerate: {res['batch_rows']}")
        for feat, val in top5b.items():
            print(f"  {feat:15s}: {val:.4f}")
        print(f"Drift Ratio (batch): {batch_drift['drift_ratio']:.1%}")
        print(f"Features in drift (batch): {batch_drift['drifted_features']}")
        print(f"should_retrain (batch PSI): {batch_drift['should_retrain']}")

    # Nota: il retrain è già stato fatto automaticamente dopo le 5 case;
    # questo script serve solo a misurare quanto i due dataset/modelli differiscono.
