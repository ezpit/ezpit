"""Demo: Compton scattering intensity calculation.

[EN] Takes a chemical composition string (required argument) and computes the
     experimental Compton scattering intensity using the built-in element tables
     (``ezpit.elem_tables``), so no external database files are required. Supports
     integer and fractional compositions.
[KR] 화학 조성 문자열(필수 인자)을 받아 내장 원소 테이블을 사용해 실험적 콤프턴
     산란 강도를 계산합니다. 정수/소수 조성을 모두 지원합니다.

Run with:
    uv run --extra examples python examples/demo_Comptonscattering_0731_2026.py \
        Li0.2Co0.36Mn0.37Ni0.07
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import ezpit.processing as proc
from ezpit.elem_tables import (
    get_compton_parameter_only,
    get_compton_scattering_factors,
)
from ezpit.io import composition_weights, parse_composition

# ----------------------------------------------------------------------------------
# Command-line arguments (명령행 인자)
# ----------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compton scattering intensity from a composition string.")
parser.add_argument(
    "composition",
    help="Chemical composition string, e.g. 'Li0.2Co0.36Mn0.37Ni0.07' or 'Co38O119P20'.",
)
parser.add_argument(
    "--wavelength", type=float, default=0.1665, help="X-ray wavelength in Angstrom (default: %(default)s)."
)
parser.add_argument("--alpha", type=int, default=3, choices=[2, 3] help="Breit-Dirac recoil parameter, 2 or 3 (default: %(default)s).")
parser.add_argument("--qmin", type=float, default=0.0, help="Minimum q in 1/A (default: %(default)s).")
parser.add_argument("--qmax", type=float, default=30.0, help="Maximum q in 1/A (default: %(default)s).")
parser.add_argument("--qstep", type=float, default=0.01, help="q step size in 1/A (default: %(default)s).")
args = parser.parse_args()

composition = args.composition
wavelength = args.wavelength  # [EN] X-ray wavelength (Angstrom) / [KR] X선 파장
alpha = args.alpha  # [EN] Breit-Dirac recoil parameter (2 or 3) / [KR] 반동 보정 계수
qmin = args.qmin
qmax = args.qmax
qstep = args.qstep

# [EN] Parse and get per-unique-element names + weights (fraction-safe).
# [KR] 파싱하여 고유 원소 이름 + weight 획득 (소수 지원).
comp_parsed = parse_composition(composition)
unique_atom_names, comp_weights = composition_weights(comp_parsed)

# [EN] Look up Compton parameters and atomic numbers from the built-in tables.
# [KR] 내장 테이블에서 Compton 파라미터와 원자 번호를 조회합니다.
compton_scat_form_factor, atomic_number = get_compton_scattering_factors(unique_atom_names)
compton_scat_parms = get_compton_parameter_only()

# [EN] atom_indices is kept for backward compatibility; when weights= is given it
#      is not used for the averaging (see compton_cal_exp docstring).
# [KR] atom_indices는 하위 호환용이며, weights=가 주어지면 평균 계산에 쓰이지 않음.
atom_indices = np.arange(len(unique_atom_names))

# ----------------------------------------------------------------------------------
# Calculation (콤프턴 산란 강도 계산)
# ----------------------------------------------------------------------------------
list_q, list_compton_scat = proc.compton_cal_exp(
    atom_indices,
    compton_scat_parms,
    compton_scat_form_factor,
    atomic_number,
    qmin=qmin,
    qmax=qmax,
    qstep=qstep,
    wavelength=wavelength,
    alpha=alpha,
    weights=comp_weights,
)

# ----------------------------------------------------------------------------------
# Plotting (결과 그래프 출력)
# ----------------------------------------------------------------------------------
plt.figure(0)
plt.plot(list_q, list_compton_scat, label=f"Compton scattering: {composition}")
plt.xlabel("q (1/A)")
plt.ylabel("Compton intensity")
plt.grid()
plt.legend()
plt.show()
