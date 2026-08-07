"""Demo: experimental S(q)/F(q)/G(r) processing from measured I(q).

[EN] Reads an experimental 2-column q, I(q) file (required argument) and runs the
     full experimental workflow: background subtraction, structure factor S(q),
     reduced structure function F(q), and G(r) via IFFT, including
     Whittaker-Henderson smoothing and the Lorch function. Atomic form factors
     and Compton parameters come from the built-in element tables
     (``ezpit.elem_tables``), so no external database files are needed. Generated
     output files are written to a temporary directory.

     Synthetic experimental/background files can be produced with
     ``generate_example_data.py``.
[KR] 실험 2열 q, I(q) 파일(필수 인자)을 읽어 전체 실험 워크플로우를 수행합니다:
     배경 제거, S(q), F(q), IFFT G(r), Whittaker-Henderson 스무딩, Lorch 함수.
     원자 form factor와 Compton 파라미터는 내장 원소 테이블에서 가져옵니다.

Run with:
    uv run --extra examples python examples/generate_example_data.py /tmp/ezpit_demo
    uv run --extra examples python examples/demo_experimental_data_testingcode_0731_2026.py \
        /tmp/ezpit_demo/synthetic_exp.chi --background /tmp/ezpit_demo/synthetic_bkg.chi
"""

import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import ezpit.processing as proc
from ezpit.elem_tables import (
    get_aff_scattering_factors,
    get_compton_parameter_only,
    get_compton_scattering_factors,
)
from ezpit.io import composition_weights, convert_atom_names, group_atoms, parse_composition

# ----------------------------------------------------------------------------------
# Command-line arguments (명령행 인자)
# ----------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Experimental S(q)/F(q)/G(r) processing from a q, I(q) file.")
parser.add_argument("exp_file", help="Path to the experimental 2-column q, I(q) file (.chi/.iq).")
parser.add_argument(
    "--background",
    help="Optional path to a 2-column q, I(q) background file. If omitted, no background is subtracted.",
)
parser.add_argument(
    "--composition",
    default="Co38O119P20",
    help="Chemical composition of the sample (default: %(default)s).",
)
args = parser.parse_args()

output_base = tempfile.mkdtemp(prefix="ezpit_exp_demo_")
print(f"Writing generated files to: {output_base}")

expqiq_data = args.exp_file
bkgqiq_data = args.background  # may be None

# ----------------------------------------------------------------------------------
# Parameters (분석 파라미터)
# ----------------------------------------------------------------------------------
composition = args.composition
qmin = 0.6
qmax = 23.0
qstep = 0.01
background_scale = 0.27
qdamp = 0.0
poly_order = 7.208
rmin = 0.0
rmax = 20.0
rstep = 0.01
wavelength = 0.1665
alpha = 3

# ----------------------------------------------------------------------------------
# [EN] Prepare the composition: unique element names + per-atom index mapping.
# [KR] 조성 준비: 고유 원소 이름 + 원자별 인덱스 매핑.
# ----------------------------------------------------------------------------------
comp_parsed = parse_composition(composition)
is_fractional = any(not float(v).is_integer() for v in comp_parsed.values())

# [EN] Fraction-safe per-unique-element names + weights (used for S(q) averaging).
# [KR] 소수 지원 고유 원소 이름 + weight (S(q) 평균용).
comp_names, comp_weights = composition_weights(comp_parsed)

if is_fractional:
    atom_unique_names = comp_names
    atom_indices = np.arange(len(comp_names))
else:
    atom_names = convert_atom_names(comp_parsed)
    atom_unique_names, atom_counts, atom_indices = group_atoms(atom_names)

# ----------------------------------------------------------------------------------
# Compton scattering intensity (콤프턴 산란 강도)
# ----------------------------------------------------------------------------------
compton_scat_form_factor, atomic_number = get_compton_scattering_factors(atom_unique_names)
compton_scat_parms = get_compton_parameter_only()
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
    weights=comp_weights if is_fractional else None,
)

# ----------------------------------------------------------------------------------
# S(q) / F(q) processing (구조 인자 계산)
# ----------------------------------------------------------------------------------
scattering_factors = get_aff_scattering_factors(atom_unique_names)

(
    q,
    Iq,
    scaled_expIq,
    list_scaled_bkgIq,
    list_Sq,
    Sq,
    Fq,
    mean_sq_fi,
    sq_mean_fi,
    polynomial_for_sq,
    normalized_intensity,
    normal_scattering_factor,
    normalization_scale,
) = proc.cal_expSq(
    atom_indices,
    scattering_factors,
    expqiq_data,
    bkgqiq_data,
    qmin=qmin,
    qmax=qmax,
    qstep=qstep,
    background_scale=background_scale,
    poly_order=poly_order,
    return_Iq=False,
    weights=comp_weights if is_fractional else None,
)
print("normalization_scale =", normalization_scale)

# ----------------------------------------------------------------------------------
# G(r): Lorch function + Whittaker-Henderson smoothing
# ----------------------------------------------------------------------------------
Fq_lorch = proc.apply_lorch_function(q, Fq)
r_lorch, Gr_lorch = proc.cal_expGr_fft_from_Fq(q, Fq_lorch, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear")

whittaker_lambda = 1000.0
order = 2
Fq_smoothed = proc.smooth_whittaker(Fq, lambda_=whittaker_lambda, order=order)
r_smooth, Gr_from_smoothFq = proc.cal_expGr_fft_from_Fq(
    q, Fq_smoothed, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear"
)
r_raw, Gr_from_rawFq = proc.cal_expGr_fft_from_Fq(q, Fq, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear")

# [EN] G(r) directly from S(q) with different high-Q padding modes.
# [KR] 서로 다른 high-Q padding 모드로 S(q)에서 바로 G(r) 계산.
r3, Gr3 = proc.cal_expGr_fft(q, Sq, rmin, rmax, rstep, pad_mode="decay", low_q_mode="linear")
r4, Gr4 = proc.cal_expGr_fft(q, Sq, rmin, rmax, rstep, pad_mode="constant", low_q_mode="linear")
r5, Gr5 = proc.cal_expGr_fft(q, Sq, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear")

# ----------------------------------------------------------------------------------
# Save intermediate results (중간 결과 저장)
# ----------------------------------------------------------------------------------
np.savetxt(os.path.join(output_base, "q_scaled_expIq.iq"), np.column_stack([q, scaled_expIq]))
np.savetxt(os.path.join(output_base, "q_list_scaled_bkgIq.iq"), np.column_stack([q, list_scaled_bkgIq]))
np.savetxt(os.path.join(output_base, "bkg_subtracted_expqIq.iq"), np.column_stack([q, Iq]))
np.savetxt(os.path.join(output_base, "notnormalized_qSq.sq"), np.column_stack([q, list_Sq]))
np.savetxt(os.path.join(output_base, "polynomial_qpolySq.sq"), np.column_stack([q, polynomial_for_sq]))
np.savetxt(os.path.join(output_base, "normalized_qSq.sq"), np.column_stack([q, Sq]))
np.savetxt(os.path.join(output_base, "qFq.fq"), np.column_stack([q, Fq]))

# ----------------------------------------------------------------------------------
# Plotting (결과 그래프 출력)
# ----------------------------------------------------------------------------------
plt.figure(0)
plt.plot(list_q, list_compton_scat, label="Compton scattering")
plt.xlabel("q (1/A)")
plt.ylabel("Compton intensity")
plt.grid()
plt.legend()

plt.figure(1)
plt.plot(q, scaled_expIq, label="Exp (raw)")
plt.plot(q, list_scaled_bkgIq, label="Bkg * scale")
plt.plot(q, Iq, label="Net I(q)")
plt.xlabel("q (1/A)")
plt.ylabel("I(q)")
plt.grid()
plt.legend()

plt.figure(3)
plt.plot(q, list_Sq, label="Standard S(q)")
plt.plot(q, Sq, label="Poly corrected S(q)")
plt.xlabel("q (1/A)")
plt.ylabel("S(q)")
plt.grid()
plt.legend()

plt.figure(4)
plt.plot(q, Fq, label="F(q)")
plt.plot(q, Fq_smoothed, label="F(q) WH-smoothed")
plt.plot(q, Fq_lorch, label="F(q) Lorch")
plt.xlabel("q (1/A)")
plt.ylabel("F(q)")
plt.grid()
plt.legend()

plt.figure(7)
plt.plot(r3, Gr3, label="decay")
plt.plot(r4, Gr4, label="constant")
plt.plot(r5, Gr5, label="zero")
plt.plot(r_lorch, Gr_lorch, label="Lorch")
plt.plot(r_smooth, Gr_from_smoothFq, label="WH-smoothed")
plt.title("G(r) comparison")
plt.xlabel("r (A)")
plt.ylabel("G(r)")
plt.grid()
plt.legend()
plt.show()
