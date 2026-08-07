"""Demo: theoretical PDF from an .xyz atomic model (Debye scattering equation).

[EN] Loads a molecular structure from an .xyz file (required argument), then
     computes I(q), S(q), F(q) and G(r) from it using the Debye scattering
     equation. Atomic form factors come from the built-in element tables
     (``ezpit.elem_tables``), so no external database files are needed. Generated
     output files are written to a temporary directory.

     A real 76-atom structure is shipped in ``examples/example_data``; a synthetic
     one can be produced with ``generate_example_data.py``.
[KR] .xyz 파일(필수 인자)에서 분자 구조를 불러온 뒤, Debye 산란 공식으로 I(q),
     S(q), F(q), G(r)를 계산합니다. 원자 form factor는 내장 원소 테이블에서
     가져오므로 외부 데이터베이스 파일이 필요 없습니다.

Run with:
    uv run --extra examples python examples/demo_xyz_model_0731_2026.py \
        examples/example_data/Iaa-Iaa_solute_0001.xyz
"""

import argparse
import os
import tempfile
import timeit

import matplotlib.pyplot as plt
import numpy as np

import ezpit.io as losa
import ezpit.processing as proc
from ezpit.elem_tables import AFF_ELEMENTS, get_aff_scattering_factors

# ----------------------------------------------------------------------------------
# Command-line arguments (명령행 인자)
# ----------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Theoretical PDF from an .xyz atomic model.")
parser.add_argument("xyz_file", help="Path to the input .xyz structure file (element x y z per line).")
parser.add_argument("--qmin", type=float, default=0.0, help="Minimum q in 1/A (default: %(default)s).")
parser.add_argument("--qmax", type=float, default=30.0, help="Maximum q in 1/A (default: %(default)s).")
parser.add_argument("--qstep", type=float, default=0.01, help="q step size in 1/A (default: %(default)s).")
parser.add_argument("--rmin", type=float, default=0.0, help="Minimum r in A (default: %(default)s).")
parser.add_argument("--rmax", type=float, default=30.0, help="Maximum r in A (default: %(default)s).")
parser.add_argument("--rstep", type=float, default=0.01, help="r step size in A (default: %(default)s).")
parser.add_argument("--qdamp", type=float, default=0.0, help="Resolution damping factor (default: %(default)s).")
args = parser.parse_args()

output_base = tempfile.mkdtemp(prefix="ezpit_xyz_demo")
print(f"Writing generated files to: {output_base}")

# ----------------------------------------------------------------------------------
# [EN] Load atom names and (x, y, z) positions from the .xyz file. Passing the
#      element list from the form-factor table as valid_symbols lets the loader
#      recognise species (including ions like 'Fe2+') and skip header lines.
# [KR] .xyz 파일에서 원자 이름과 (x, y, z) 좌표를 불러옵니다.
# ----------------------------------------------------------------------------------
atom_names, atom_positions = losa.load_atom_name_positions(args.xyz_file, AFF_ELEMENTS)

# ----------------------------------------------------------------------------------
# Calculation parameters (계산 파라미터)
# ----------------------------------------------------------------------------------
qmin = args.qmin
qmax = args.qmax
qstep = args.qstep
rmin = args.rmin
rmax = args.rmax
rstep = args.rstep
qdamp = args.qdamp  # [EN] Resolution damping (0 for theory) / [KR] 해상도 감쇠 (이론 계산은 보통 0)

t_start = timeit.default_timer()

# [EN] Pairwise distance matrix between all atoms (needed by the Debye formula).
# [KR] 모든 원자 쌍 사이의 거리 행렬 (Debye 공식에 필요).
atom_distance_matrix = proc.create_atom_distance_matrix(atom_positions)

# [EN] Group atoms by element. cal_Sq looks up scattering factors by unique-element
#      index, so scattering_factors must be one row per unique element in the same
#      order returned by group_atoms.
# [KR] 원자를 원소별로 그룹화. cal_Sq는 고유 원소 인덱스로 산란 인자를 조회하므로
#      scattering_factors는 group_atoms가 반환한 순서대로 고유 원소당 한 행이어야 함.
atom_unique_names, atom_counts, atom_indices = losa.group_atoms(atom_names)

# [EN] Atomic form factors for the unique elements (from the built-in tables).
# [KR] 고유 원소들의 원자 form factor (내장 테이블에서).
scattering_factors = get_aff_scattering_factors(atom_unique_names)

# [EN] Theoretical I(q), S(q), F(q) via the Debye scattering equation.
# [KR] Debye 산란 공식으로 이론적 I(q), S(q), F(q) 계산.
q, Iq, Sq, Fq, mean_sq_fi, sq_mean_fi = proc.cal_Sq(
    atom_indices,
    scattering_factors,
    atom_distance_matrix,
    qmin=qmin,
    qmax=qmax,
    qstep=qstep,
    return_Iq=True,
)

# ----------------------------------------------------------------------------------
# Plot I(q), S(q), F(q)
# ----------------------------------------------------------------------------------
plt.figure(0)
plt.plot(q, Iq, label="I(q)")
np.savetxt(os.path.join(output_base, "model.iq"), np.column_stack([q, Iq]))
plt.xlabel("q (1/A)")
plt.ylabel("I(q)")
plt.yscale("log")
plt.grid()
plt.legend()

plt.figure(1)
plt.plot(q, Sq, label="S(q)")
plt.xlabel("q (1/A)")
plt.ylabel("S(q)")
plt.grid()
plt.legend()

plt.figure(2)
plt.plot(q, Fq, label="F(q)")
plt.xlabel("q (1/A)")
plt.ylabel("F(q)")
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Method 1: integral transform (slow but exact) — 적분 변환 (정확함)
# ----------------------------------------------------------------------------------
t0 = timeit.default_timer()
r, Gr = proc.cal_Gr_integral(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=qdamp)
np.savetxt(os.path.join(output_base, "model_integral.gr"), np.column_stack([r, Gr]))
print("G(r) integral time cost:", timeit.default_timer() - t0)

plt.figure(3)
plt.plot(r, Gr, label="integral")

# ----------------------------------------------------------------------------------
# Method 2: IFFT (fast) — 고속 푸리에 변환 (빠름)
# ----------------------------------------------------------------------------------
t0 = timeit.default_timer()
r, Gr2 = proc.cal_Gr_fft(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=qdamp, extrapolate_type="linear")
np.savetxt(os.path.join(output_base, "model_ifft.gr"), np.column_stack([r, Gr2]))
print("G(r) IFFT time cost:", timeit.default_timer() - t0)
print("Total time cost:", timeit.default_timer() - t_start)

plt.plot(r, Gr2, label="ifft")
plt.xlabel("r (A)")
plt.ylabel("G(r)")
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Comparison: integral vs IFFT (0에 가까울수록 좋음)
# ----------------------------------------------------------------------------------
plt.figure(4)
plt.plot(r, Gr - Gr2, label="difference (integral - ifft)")
plt.xlabel("r (A)")
plt.ylabel("G(r) - G(r)2")
plt.grid()
plt.legend()
plt.show()
