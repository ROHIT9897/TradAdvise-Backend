# diagnose.py — replace the full file
import asyncio
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())

async def diagnose():
    print("=== DIAGNOSIS ===\n")

    # ── Step 1: Fetch data ───────────────────────────
    print("1. FETCHING DATA:")
    from data.fetcher import get_historical_data
    try:
        df = await get_historical_data("RELIANCE", period="2y")
        print(f"   ✓ Rows: {len(df)}")
        print(f"   ✓ Columns: {df.columns.tolist()}")
        print(f"   ✓ NaN count: {df.isnull().sum().sum()}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        return

    # ── Step 2: Build features with full traceback ───
    print("\n2. BUILDING FEATURES:")
    try:
        from ml.features import build_features
        features = build_features(df)

        if features is None:
            print("   ✗ build_features() returned None")
            print("   → Check your features.py — missing return statement?")
            return

        print(f"   ✓ Shape: {features.shape}")
        print(f"   ✓ Columns: {features.columns.tolist()}")

        if 'target' not in features.columns:
            print("   ✗ 'target' column missing from features")
            return

        dist = features['target'].value_counts()
        total = len(features)
        print(f"\n   CLASS DISTRIBUTION:")
        for label, count in dist.items():
            name = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}.get(label, str(label))
            print(f"   {name}: {count} ({count/total*100:.1f}%)")

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        return

    # ── Step 3: Feature correlation ──────────────────
    print("\n3. TOP FEATURES CORRELATED WITH TARGET:")
    try:
        X = features.drop('target', axis=1)
        y = features['target']
        corr = X.corrwith(y).abs().sort_values(ascending=False)
        print(corr.head(10).to_string())
        print(f"\n   Best correlation: {corr.iloc[0]:.4f}")
        if corr.iloc[0] < 0.15:
            print("   ⚠️  Very weak — features not predictive of target")
        elif corr.iloc[0] < 0.25:
            print("   ⚠️  Moderate — model will struggle")
        else:
            print("   ✓ Good correlation — model should work")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Step 4: Baseline check ───────────────────────
    print("\n4. BASELINE CHECK:")
    try:
        majority_pct = dist.max() / total * 100
        majority_label = {1:'BUY', 0:'HOLD', -1:'SELL'}.get(dist.idxmax(), '?')
        print(f"   Always predict {majority_label}: {majority_pct:.1f}%")
        print(f"   Your model needs to beat this to be useful")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ── Step 5: Check features.py return ─────────────
    print("\n5. CHECKING features.py FILE:")
    try:
        with open("ml/features.py", "r") as f_file:
            content = f_file.read()
            lines = content.split('\n')
            # Find return statements
            returns = [(i+1, l.strip()) for i, l in enumerate(lines) if 'return' in l]
            print(f"   Return statements found: {len(returns)}")
            for lineno, line in returns:
                print(f"   Line {lineno}: {line}")
    except Exception as e:
        print(f"   ✗ Could not read features.py: {e}")

    print("\n=== DONE ===")

asyncio.run(diagnose())