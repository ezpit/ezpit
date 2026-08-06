# ----------------------------------------------------------------------------------
# [EN] Path setup: add the EZPDF_code_version folder (parent of 'examples') to
#      sys.path so 'losa' and 'proc' packages can be imported regardless of the
#      current working directory.
# [KR] 경로 설정: 'examples'의 상위 폴더(EZPDF_code_version)를 sys.path에 추가하여
#      실행 위치와 무관하게 'losa', 'proc' 패키지를 import할 수 있게 합니다.
# ----------------------------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timeit  # [EN] Tool to measure execution time / [KR] 코드 실행 시간을 측정하는 도구

import matplotlib.pyplot as plt  # [EN] Library for plotting graphs / [KR] 그래프를 그리기 위한 라이브러리
import numpy as np  # [EN] Library for numerical array processing / [KR] 수치 계산 및 배열 처리를 위한 라이브러리

import ezpit.io as losa  # [EN] Custom module for loading/saving data / [KR] 데이터 로드 및 저장을 위한 사용자 정의 모듈

# [EN] Custom module for scientific calculations
# [KR] 과학적 계산(S(q), G(r) 등)을 위한 사용자 정의 모듈
import ezpit.processing as proc

# [EN] Define file paths (String)
# [KR] 파일 경로 설정 (문자열 변수)
input_base = "C:/Users/gkwon/Pycharmprojects/ezpit/EZPDF_code_version/data/"

# [EN] Input files: G(r) experimental data (for comparison) and XYZ structure file
# [KR] 입력 파일: 비교용 G(r) 데이터와 이론적 계산을 위한 구조 파일(.xyz)
input_exp_file = input_base + "/sum_A_CoPiITO_110320-1_Nsum5.chi_integral.gr"
atom_xyz_file = (
    input_base + "/Iaa-Iaa_solute_0001.xyz"
)  # Iaa-Iaa_solute_0001.xyz" #5IrC_r5a-1Ir.xyz" #5IrC_r5a-1Ir.xyz"  #Ni(OH)2-109391-ICSD-10x10x1.xyz"

# [EN] Database files for atomic scattering factors
# [KR] 원자 산란 인자 정보를 담은 데이터베이스 파일 경로
aff_element_file = input_base + "/aff_elementonly.txt"
aff_parm_file = input_base + "aff_parmonly.txt"

# [EN] Load atomic database FIRST (element/ion names + scattering parameters),
#      so the element/ion list can validate the .xyz species (including IONS).
# [KR] 원자 데이터베이스를 먼저 로드합니다 (원소/이온 기호 + 산란 파라미터).
#      이 원소/이온 목록으로 .xyz의 화학종을 검증합니다 (이온 포함).
database_atom_names = losa.load_atom_names(aff_element_file)
database_scat_factors = losa.load_scattering_factors(aff_parm_file)

# [EN] Load atom names and (x,y,z) positions from .xyz file.
#      Passing database_atom_names as valid_symbols lets the loader recognise
#      IONS present in the form-factor table (e.g. 'Fe2+', 'O2-', 'Cl1-'), and
#      auto-skip any header/comment lines.
# [KR] .xyz 파일에서 원자 이름 리스트와 (x,y,z) 좌표 배열을 불러옵니다.
#      database_atom_names를 valid_symbols로 넘기면 form-factor 테이블에 있는
#      이온('Fe2+', 'O2-', 'Cl1-' 등)도 인식하고 헤더/주석 줄은 자동으로 건너뜁니다.
# atom_names: List of strings ['Ir', 'Ir', ...] (or ions like ['Fe2+', ...])
# atom_positions: Numpy array (N rows, 3 columns)
atom_names, atom_positions = losa.load_atom_name_positions(atom_xyz_file, database_atom_names)
# print(atom_names)

# import sys
# sys.exit(0)

# print(atom_names)
# print(atom_positions)
# print(len(database_atom_names))
# print(database_scat_factors.shape)

# [EN] Calculation Parameters (Simulation settings)
# [KR] 계산 파라미터 설정 (시뮬레이션 환경 설정)
qmin = 0.0  # [EN] Minimum Q value (1/A) / [KR] Q 최소값
qmax = 30  # [EN] Maximum Q value (1/A) / [KR] Q 최대값
qstep = 0.01  # [EN] Step size for Q / [KR] Q 간격 (작을수록 정밀)
rmin = 0  # [EN] Minimum r value (A) / [KR] 거리 r 최소값
rmax = 30  # [EN] Maximum r value (A) / [KR] 거리 r 최대값
rstep = 0.001  # [EN] Step size for r / [KR] 거리 r 간격
qdamp = 0.0  # [EN] Resolution damping factor / [KR] 기기 해상도 감쇠 인자 (이론 계산시 보통 0)
# method = 'ifft'

# [EN] Start timer to measure total performance
# [KR] 전체 성능 측정을 위해 타이머 시작
t5 = timeit.default_timer()

# [EN] Calculate pairwise distance matrix between all atoms
# [KR] 모든 원자 쌍 사이의 거리 행렬을 계산합니다. (Debye 공식을 위해 필수)
# Output: 2D Numpy array (N x N)
atom_distance_matrix = proc.create_atom_distance_matrix(atom_positions)

# [EN] Group atoms by element type
# [KR] 원자들을 종류별로 분류하고 개수를 셉니다.
atom_unique_names, atom_counts, atom_indices = losa.group_atoms(atom_names)

# [EN] Retrieve scattering factors for the atoms in the structure
# [KR] 구조에 포함된 원자들의 산란 인자 계수를 가져옵니다.
scattering_factors = losa.get_scattering_factors(atom_unique_names, database_atom_names, database_scat_factors)

# [EN] Calculate theoretical I(q), S(q), F(q) using the Debye scattering equation
# [KR] Debye 산란 공식을 사용하여 이론적인 I(q), S(q), F(q)를 계산합니다.
# q: x-axis, Iq: Scattering Intensity, Sq: Structure Factor
q, Iq, Sq, Fq, mean_sq_fi, sq_mean_fi = proc.cal_Sq(
    atom_indices,
    scattering_factors,
    atom_distance_matrix,
    qmin=qmin,
    qmax=qmax,
    qstep=qstep,
    return_Iq=True,
)

# print(qIq.shape)

# ----------------------------------------------------------------------------------
# Plotting I(q) - Intensity
# ----------------------------------------------------------------------------------
plt.figure(0)
plt.plot(q, Iq, label="I(q)")

# [EN] Save calculated I(q) to a text file
# [KR] 계산된 I(q) 데이터를 텍스트 파일로 저장합니다.
np.savetxt(input_base + "5IrC_r5a-1Ir_integral_qmin0p5.iq", np.column_stack([q, Iq]))

# plt.plot(q, sq_mean_fi, label='sq_mean_fi')
# plt.plot(q, mean_sq_fi, label='mean_sq_fi')
plt.xlabel("q (1/A)")
plt.ylabel("I(q)")
# plt.xscale("log")
plt.yscale("log")  # [EN] Log scale for Y-axis / [KR] Y축을 로그 스케일로 설정
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Plotting S(q) - Structure Factor
# ----------------------------------------------------------------------------------
plt.figure(1)
plt.plot(q, Sq, label="S(q)")
plt.xlabel("q (1/A)")
plt.ylabel("S(q)")
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Plotting F(q) - Reduced Structure Function
# ----------------------------------------------------------------------------------
plt.figure(2)
plt.plot(q, Fq, label="F(q)")
plt.xlabel("q (1/A)")
plt.ylabel("F(q)")
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Method 1: Integral Transform (Slow but exact)
# [KR] 방법 1: 적분 변환 (속도는 느리지만 정확함, Integral)
# ----------------------------------------------------------------------------------
t0 = timeit.default_timer()  # [EN] Start timer for integral method / [KR] 적분 방식 타이머 시작

# [EN] Calculate G(r) using sine integral transform
# [KR] 사인 적분 변환을 사용하여 G(r) 계산
r, Gr = proc.cal_Gr_integral(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=qdamp)

# [EN] Save Integral G(r) result
# [KR] 적분법으로 계산된 G(r) 저장
np.savetxt(input_base + "5IrC_r5a-1Ir_integral_qmin0p5.gr", np.column_stack([r, Gr]))  # or use "list(zip(r, Gr)))"

t1 = timeit.default_timer()  # [EN] Stop timer / [KR] 타이머 종료
print("Time cost real space!!! ", t1 - t0)


plt.figure(3)
plt.plot(r, Gr, label="integral")  # [EN] Plot Integral result / [KR] 적분 결과 플롯
plt.xlabel("r (A)")
plt.ylabel("G(r)")
plt.legend()

# ----------------------------------------------------------------------------------
# Method 2: IFFT (Fast Fourier Transform) (Fast)
# [KR] 방법 2: 고속 푸리에 변환 (IFFT) (속도가 빠름)
# ----------------------------------------------------------------------------------
t0 = timeit.default_timer()  # [EN] Start timer for IFFT method / [KR] FFT 방식 타이머 시작

# [EN] Calculate G(r) using IFFT
# [KR] IFFT를 사용하여 G(r) 계산
r, Gr2 = proc.cal_Gr_fft(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=qdamp, extrapolate_type="linear")

# [EN] Save IFFT G(r) result
# [KR] IFFT로 계산된 G(r) 저장
np.savetxt(input_base + "5IrC_r5a-1Ir_ifft_qmin0p5.gr", np.column_stack([r, Gr2]))  # or use "list(zip(r, Gr)))"

t1 = timeit.default_timer()
print("Time cost Fourier space!!! ", t1 - t0)
t6 = timeit.default_timer()

print("Total time cost including real and Fourier space!!! ", t6 - t5)

plt.plot(r, Gr2, label="ifft")  # [EN] Plot IFFT result / [KR] IFFT 결과 플롯
plt.xlabel("r (A)")
plt.ylabel("G(r)")
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Comparison: Difference between Integral and IFFT
# [KR] 비교: 적분법과 FFT법의 차이 확인 (0에 가까울수록 좋음)
# ----------------------------------------------------------------------------------
plt.figure(4)
plt.plot(r, Gr - Gr2, label="difference")
plt.xlabel("r (A)")
plt.ylabel("G(r)-G(r)2")
plt.grid()
plt.legend()
plt.show()
