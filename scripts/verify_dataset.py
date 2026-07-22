#!/usr/bin/env python3
"""Verify the extracted KdV dataset without retraining."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REQUIRED = {
    "data/processed/LAB_E1_core_archive.npz",
    "data/processed/COMOV_E1_core_archive.npz",
    "data/processed/COMOV_E1_residual_diag.npz",
    "data/models/LAB_E1_model_bundle.npz",
    "data/models/COMOV_E1_model_bundle.npz",
    "data/models/LAB_E1_net1_stage1.pt",
    "data/models/LAB_E1_net2_stage2.pt",
    "data/models/COMOV_E1_net1_stage1.pt",
    "data/models/COMOV_E1_net2_stage2.pt",
    "data/source_data/Figure1_source_data.csv",
    "data/source_data/Figure2_source_data.csv",
    "data/source_data/Figure3_source_data.csv",
    "data/source_data/Figure4_source_data.npz",
    "data/source_data/Figure5_source_data.npz",
    "data/source_data/Figure6_source_data.npz",
    "data/source_data/Supplementary_Figure_S1_source_data.npz",
    "data/source_data/Supplementary_Figure_S2_source_data.npz",
    "data/source_data/Supplementary_Figure_S3_source_data.npz",
}

EXPECTED_KEYS = {
    "data/processed/LAB_E1_core_archive.npz": {
        "tau_ref", "xi_ref", "u_ref", "U2", "relL2_stage1", "relL2_stage2",
        "crest_shift_stage1", "crest_shift_stage2", "dI1_stage1", "dI2_stage1",
        "dI3_stage1", "dI1_stage2", "dI2_stage2", "dI3_stage2",
    },
    "data/processed/COMOV_E1_core_archive.npz": {
        "tau_ref", "xi_ref", "eta_ref", "u_ref", "U_ref", "U2",
        "relL2_stage1", "relL2_stage2", "crest_shift_stage1", "crest_shift_stage2",
        "dI1_stage1", "dI2_stage1", "dI3_stage1", "dI1_stage2", "dI2_stage2", "dI3_stage2",
    },
    "data/processed/COMOV_E1_residual_diag.npz": {
        "times", "eta", "U", "Rabs", "Den", "Rho", "Rho_masked",
        "term_t_magnitude", "term_nl_magnitude", "term_disp_magnitude",
    },
    "data/source_data/Figure6_source_data.npz": {
        "times", "eta", "term_t_magnitude", "term_nl_magnitude",
        "term_disp_magnitude", "residual_magnitude",
        "normalization_denominator", "normalized_residual_ratio",
    },
}

EXPECTED = {
    "comov_relL2_final": ("data/processed/COMOV_E1_core_archive.npz", "relL2_stage2", "last", 1.5002965e-3),
    "comov_crest_max": ("data/processed/COMOV_E1_core_archive.npz", "crest_shift_stage2", "maxabs", 2.1446194e-3),
    "comov_invariant_envelope": ("data/processed/COMOV_E1_core_archive.npz", "invariants_stage2", "maxabs", 5.691356e-4),
    "space_time_error_max": ("data/source_data/Figure5_source_data.npz", "abs_err_stage2", "max", 4.2966008e-4),
}


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing data/")
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    print(f"Project root: {root}")
    for rel in sorted(REQUIRED):
        p = root / rel
        if not p.is_file():
            failures.append(f"Missing required file: {rel}")
    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        return 1
    print(f"[PASS] Required file inventory ({len(REQUIRED)} files)")

    for rel, needed in EXPECTED_KEYS.items():
        with np.load(root / rel, allow_pickle=False) as z:
            missing = sorted(needed - set(z.files))
            if missing:
                failures.append(f"{rel}: missing keys {missing}")
            for key in z.files:
                arr = np.asarray(z[key])
                if np.issubdtype(arr.dtype, np.number) and key != "Rho_masked":
                    if not np.isfinite(arr).all():
                        failures.append(f"{rel}:{key} contains non-finite values")
    print("[PASS] Required NPZ schemas and finite-value checks" if not failures else "[FAIL] NPZ schema check")

    for rel in sorted(p for p in REQUIRED if p.endswith('.pt')):
        obj = torch.load(root / rel, map_location='cpu', weights_only=True)
        if not isinstance(obj, dict) or not obj:
            failures.append(f"{rel}: did not load as a non-empty state dictionary")
    print("[PASS] PyTorch state dictionaries load with weights_only=True" if not failures else "[FAIL] Model check")

    observed = {}
    with np.load(root / EXPECTED['comov_relL2_final'][0], allow_pickle=False) as z:
        observed['comov_relL2_final'] = float(z['relL2_stage2'][-1])
        observed['comov_crest_max'] = float(np.max(np.abs(z['crest_shift_stage2'])))
        observed['comov_invariant_envelope'] = float(max(
            np.max(np.abs(z['dI1_stage2'])),
            np.max(np.abs(z['dI2_stage2'])),
            np.max(np.abs(z['dI3_stage2'])),
        ))
    with np.load(root / EXPECTED['space_time_error_max'][0], allow_pickle=False) as z:
        observed['space_time_error_max'] = float(np.max(z['abs_err_stage2']))

    for name, value in observed.items():
        reference = EXPECTED[name][3]
        rd = rel_diff(value, reference)
        status = 'PASS' if rd <= 5e-4 else 'FAIL'
        print(f"[{status}] {name}: observed={value:.10e}, reference={reference:.10e}, relative_difference={rd:.3e}")
        if status == 'FAIL':
            failures.append(f"Metric mismatch: {name}")

    if failures:
        print("\nOverall: FAIL")
        for item in failures:
            print(f" - {item}")
        return 1
    print("\nOverall: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
