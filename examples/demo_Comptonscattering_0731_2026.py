# ----------------------------------------------------------------------------------
# [EN] Path setup: add the EZPDF_code_version folder (parent of 'examples') to
#      sys.path so 'losa' and 'proc' packages can be imported regardless of the
#      current working directory.
# [KR] 경로 설정: 'examples'의 상위 폴더(EZPDF_code_version)를 sys.path에 추가하여
#      실행 위치와 무관하게 'losa', 'proc' 패키지를 import할 수 있게 합니다.
# ----------------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # [EN] Library for plotting graphs / [KR] 그래프를 그리기 위한 라이브러리
import losa.loadersaver as losa  # [EN] Custom module for loading/saving data / [KR] 데이터 로드/저장용 사용자 정의 모듈
import proc.processing as proc   # [EN] Custom module for scientific calculations / [KR] 과학적 계산용 사용자 정의 모듈
import timeit                    # [EN] Tool to measure execution time / [KR] 실행 시간 측정 도구
import numpy as np               # [EN] Library for numerical array processing / [KR] 수치 계산 및 배열 처리를 위한 라이브러리

# [EN] Set base directory for data files
# [KR] 데이터 파일들이 위치한 기본 경로 설정
# [Type: str] (문자열)
input_base = "C:/Users/gkwon/Pycharmprojects/ezpit_local/data/"

# [EN] File paths for Compton scattering databases
# [KR] 콤프턴 산란 계산에 필요한 데이터베이스 파일 경로들
# [Type: str] (문자열)
compton_aff_element_file = input_base + 'compton_element_only.txt'   # 원소 기호 목록
compton_aff_parm_file = input_base + 'compton_parameter_only.txt'    # 산란 파라미터 목록
compton_atomnumber_file = input_base + 'compton_atomicnumber.txt'    # 원자 번호 목록

# [EN] Parameters for Calculation
# [KR] 계산에 필요한 물리적/실험적 파라미터 설정
# [Type: float] X-ray wavelength in Angstroms (X선 파장)
wavelength = 0.1665
# [Type: int] Breit-Dirac recoil parameter (usually 2 or 3) (반동 보정 계수)
alpha = 3  #2 or 3 can be used.
# ----------------------------------------------------------------------------------
# [EN] Chemical composition of the sample — EZPDF_GUI_3 compatible.
#      Accepted formats:
#        (1) dict           : {'C': 30, 'H': 24, 'N': 6, 'Ru': 1}
#        (2) compact string : "C30H24N6Ru1"  or  "C30H24N6Ru"
#        (3) spaced string  : "C 30 H 24 N 6 Ru 1"
#        (4) count-1 omitted: "SiO2"   (== {'Si': 1, 'O': 2})
#        (5) fractions      : "Li0.2Co0.36Mn0.37Ni0.07" or {'C':0.3,'H':0.24,...}
#      Fractional compositions are handled AUTOMATICALLY below (scaled to the
#      smallest whole numbers for the Compton atom list; result is identical).
# [KR] 샘플의 화학 조성 — EZPDF_GUI_3 호환. 지원 형식:
#        (1) 딕셔너리     : {'C': 30, 'H': 24, 'N': 6, 'Ru': 1}
#        (2) 붙여쓴 문자열 : "C30H24N6Ru1"  또는  "C30H24N6Ru"
#        (3) 공백 문자열   : "C 30 H 24 N 6 Ru 1"
#        (4) 개수 1 생략   : "SiO2"   (== {'Si': 1, 'O': 2})
#        (5) 소수 조성     : "Li0.2Co0.36Mn0.37Ni0.07" 또는 {'C':0.3,'H':0.24,...}
#      소수 조성은 아래에서 자동으로 처리됩니다 (Compton 원자 리스트를 위해
#      최소 정수배로 스케일; 결과는 동일).
# ----------------------------------------------------------------------------------
# composition = {'Co':2, 'O':2, 'P':1}
# composition = {'C': 30, 'H': 24, 'N': 6, 'Ru': 1}  # [EN] dict form / [KR] 딕셔너리 형식
#composition = "Li0.2Co0.36Mn0.37Ni0.07"
# composition = {'C':0.3, 'H': 0.24, 'N':0.06, 'Ru': 0.01}
# composition = "C30H24N6Ru1"                       # [EN] String form (same result) / [KR] 문자열 (동일 결과)
# composition = "C 30 H 24 N 6 Ru 1"                # [EN] Spaced string / [KR] 공백 구분 문자열
# composition = {'C': 0.30, 'H': 0.24, 'N': 0.06, 'Ru': 0.01}  # [EN] Fractional dict (auto-handled) / [KR] 소수 딕셔너리 (자동 처리)
composition = "Li0.2Co0.36Mn0.37Ni0.07Fe2+0.1"          # [EN] Fractional string (auto-handled) / [KR] 소수 문자열 (자동 처리)
# [Type: float] Q range settings (Q 최소값, 최대값, 간격)
qmin = 0
qmax = 30
qstep = 0.01

# [EN] Parse the composition and detect integer vs fractional.
#      - Integer  → build a per-atom name list (atom_indices path).
#      - Fractional → keep the exact fractions as per-element weights and pass
#        weights= to compton_cal_exp (Li0.2Co0.36... used directly, NOT scaled).
#      Both give identical Compton intensity.
# [KR] 조성을 파싱하고 정수/소수를 감지합니다.
#      - 정수  → 원자별 이름 리스트 생성 (atom_indices 경로).
#      - 소수  → 소수를 원소별 weight로 그대로 유지하고 compton_cal_exp에
#        weights=로 전달 (Li0.2Co0.36... 그대로 사용, 정수배 변환 안 함).
#      두 방식 모두 동일한 Compton 강도를 줍니다.
comp_parsed = losa.parse_composition(composition)   # dict (values may be float)
is_fractional = any(not float(v).is_integer() for v in comp_parsed.values())

if is_fractional:
    # [EN] Fractional: unique element names + exact fractional weights.
    # [KR] 소수: 고유 원소 이름 + 정확한 소수 weight.
    atom_unique_names, comp_weights = losa.composition_weights(comp_parsed)
    atom_names = None      # [EN] not needed on the fractional path / [KR] 소수 경로에선 불필요
else:
    # [EN] Integer: expand to a per-atom name list, then group.
    # [KR] 정수: 원자별 이름 리스트로 확장 후 그룹화.
    atom_names = losa.convert_atom_names(comp_parsed)
    comp_weights = None

# -----------------------------------------------------------------------------
# Database Loading
# -----------------------------------------------------------------------------
# [EN] Load all atom names and parameters from the database files
# [KR] 데이터베이스 파일에서 모든 원자 이름과 콤프턴 산란 파라미터를 불러옵니다.
# compton_atom_names: [Type: list of str] (DB에 있는 모든 원소 기호)
# compton_scat_parms: [Type: numpy.ndarray] (DB에 있는 모든 파라미터)
compton_atom_names = losa.load_atom_names(compton_aff_element_file)
compton_scat_parms = losa.load_scattering_factors(compton_aff_parm_file)

# [EN] Group atoms to find unique elements and their counts (integer path only).
#      For the fractional path, atom_unique_names is already set above.
# [KR] 원자를 종류별로 그룹화하여 고유 원소와 개수를 찾습니다 (정수 경로).
#      소수 경로에서는 atom_unique_names가 위에서 이미 설정되었습니다.
# atom_unique_names: [Type: list] Unique elements (e.g., ['Co', 'O', 'P'])
# atom_counts: [Type: numpy.ndarray] Counts per element (e.g., [2, 2, 1])
# atom_indices: [Type: numpy.ndarray] Mapping indices (인덱스 매핑)
if is_fractional:
    # [EN] atom_unique_names already from composition_weights; no atom_indices.
    # [KR] atom_unique_names는 composition_weights에서 이미 얻음; atom_indices 없음.
    atom_counts = None
    atom_indices = None
else:
    atom_unique_names, atom_counts, atom_indices = losa.group_atoms(atom_names)

# print('atom_unique_names = ',atom_unique_names)
# print('atom_counts = ',atom_counts)
# print('atom_indices = ',atom_indices)

# [EN] Retrieve specific Compton scattering parameters for the sample's composition
# [KR] 샘플에 포함된 원소들에 해당하는 콤프턴 산란 파라미터만 추출합니다.
# both (get_scattering_factors or get_compton_scattering_factors) are working
#compton_scat_form_factor, atomic_number = losa.get_scattering_factors(atom_unique_names,
#                                                           compton_atom_names,compton_scat_parms) #[1]

# compton_scat_form_factor: [Type: numpy.ndarray] Parameters for Co, O, P
# atomic_number: [Type: list of int] Atomic numbers (e.g., [27, 8, 15])
compton_scat_form_factor, atomic_number = losa.get_compton_scattering_factors(atom_unique_names,
                                                                    compton_atom_names,compton_scat_parms) #[1]

# -----------------------------------------------------------------------------
# Calculation
# -----------------------------------------------------------------------------
# [EN] Calculate the Compton scattering intensity.
#      Fractional composition → pass weights= (exact fractions used directly).
#      Integer composition    → pass atom_indices (the usual per-atom path).
#      Both give identical Compton intensity.
# [KR] 콤프턴 산란 강도를 계산합니다.
#      소수 조성 → weights= 전달 (소수를 그대로 사용).
#      정수 조성 → atom_indices 전달 (기존 원자별 경로).
#      두 방식 모두 동일한 Compton 강도를 줍니다.
# list_q: [Type: numpy.ndarray] Q values (X-axis)
# list_compton_scat: [Type: list or numpy.ndarray] Compton intensity (Y-axis)
if is_fractional:
    list_q, list_compton_scat = proc.compton_cal_exp(
        None, compton_scat_parms, compton_scat_form_factor,
        atomic_number, qmin=qmin, qmax=qmax, qstep=qstep,
        wavelength=wavelength, alpha=alpha, weights=comp_weights)
else:
    list_q, list_compton_scat = proc.compton_cal_exp(
        atom_indices, compton_scat_parms, compton_scat_form_factor,
        atomic_number, qmin=qmin, qmax=qmax, qstep=qstep,
        wavelength=wavelength, alpha=alpha)    #proc.compton_calc_exp(XXXX)

# [EN] Save the result to a text file
# [KR] 계산 결과를 텍스트 파일로 저장합니다. (컬럼 1: Q, 컬럼 2: Intensity)
np.savetxt(input_base + 'list_compton_scat.chi', np.column_stack(([list_q, list_compton_scat]))) # or use "list(zip(r, Gr)))"

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.figure(0)
plt.plot(list_q, list_compton_scat, label='Compton_scat_pattern')
plt.grid() # [EN] Show grid / [KR] 격자 표시
plt.legend() # [EN] Show legend / [KR] 범례 표시
plt.show() # [EN] Display plot / [KR] 그래프 출력