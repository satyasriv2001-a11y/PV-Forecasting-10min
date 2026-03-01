#!/usr/bin/env python3
"""
Patch train/train_ml.py in the cloned repo to allow XGBoost to run on CPU when GPU
is not available (e.g. Colab CPU runtime or XGBoost built without GPU support).

Run from repo root after clone:
  python patch_xgb_cpu_for_colab.py
  python patch_xgb_cpu_for_colab.py /content/PV-Forecasting-10min
"""
import os
import re
import sys


def patch_train_ml(repo_root: str) -> bool:
    train_ml_path = os.path.join(repo_root, "train", "train_ml.py")
    if not os.path.isfile(train_ml_path):
        print(f"[SKIP] {train_ml_path} not found")
        return False

    with open(train_ml_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # 1) Replace raise when XGBoost GPU not available with warning and CPU fallback.
    content = re.sub(
        r'raise\s+(?:Exception|ValueError|RuntimeError)\s*\(\s*["\']?XGBoost\s+training\s+failed:\s*XGBoost\s+GPU\s+not\s+available!\s*Cannot\s+train\.?["\']?\s*\)',
        'import warnings\n    warnings.warn("XGBoost GPU not available; using CPU (slower).")\n    xgb_use_gpu = False',
        content,
        count=1,
    )
    if content == original:
        content = re.sub(
            r'raise\s+\w+\s*\(\s*f?\s*["\']XGBoost\s+training\s+failed:\s*XGBoost\s+GPU\s+not\s+available!\s*Cannot\s+train\.?["\']\s*\)',
            'import warnings\n    warnings.warn("XGBoost GPU not available; using CPU (slower).")\n    xgb_use_gpu = False',
            content,
            count=1,
        )
    if content == original:
        content = content.replace(
            'raise Exception("XGBoost training failed: XGBoost GPU not available! Cannot train.")',
            'import warnings\n    warnings.warn("XGBoost GPU not available; using CPU (slower).")\n    xgb_use_gpu = False',
        )
    if content == original:
        content = content.replace(
            "raise Exception('XGBoost training failed: XGBoost GPU not available! Cannot train.')",
            'import warnings\n    warnings.warn("XGBoost GPU not available; using CPU (slower).")\n    xgb_use_gpu = False',
        )

    # 2) Set default xgb_use_gpu = True for XGB block so GPU path still works.
    if "xgb_use_gpu = False" in content and "xgb_use_gpu = True" not in content:
        content = re.sub(
            r"(\s*if\s+config\s*\[\s*[\'\"]model[\'\"]\s*\]\s*==\s*[\'\"]XGB[\'\"]\s*:)\s*\n",
            r"\1\n    xgb_use_gpu = True  # set False when GPU unavailable\n",
            content,
            count=1,
        )

    # 3) Use CPU params when GPU not available.
    content = re.sub(
        r"([\'\"]tree_method[\'\"]\s*:\s*)[\'\"]gpu_hist[\'\"]",
        r"\1'gpu_hist' if xgb_use_gpu else 'hist'",
        content,
    )
    content = re.sub(
        r"tree_method\s*=\s*[\'\"]gpu_hist[\'\"]",
        "tree_method = 'gpu_hist' if xgb_use_gpu else 'hist'",
        content,
    )
    content = re.sub(
        r"([\'\"]device[\'\"]\s*:\s*)[\'\"]cuda[\'\"]",
        r"\1'cuda' if xgb_use_gpu else 'cpu'",
        content,
    )
    content = re.sub(
        r"device\s*=\s*[\'\"]cuda[\'\"]",
        "device = 'cuda' if xgb_use_gpu else 'cpu'",
        content,
    )

    if content == original:
        print("[WARN] No substitutions made; train_ml.py may use different patterns.")
        return False

    with open(train_ml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Patched {train_ml_path} for XGBoost CPU fallback")
    return True


def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    patch_train_ml(repo_root)


if __name__ == "__main__":
    main()
