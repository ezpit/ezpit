import matplotlib.pyplot as plt  # [EN] Library for plotting graphs / [KR] 그래프를 그리기 위한 라이브러리
import ezpit.io as losa  # [EN] Custom module for loading/saving data / [KR] 데이터 로드/저장용 사용자 정의 모듈
import ezpit.processing as proc   # [EN] Custom module for scientific calculations / [KR] 과학적 계산용 사용자 정의 모듈
import timeit                    # [EN] Tool to measure execution time / [KR] 실행 시간 측정 도구
import numpy as np               # [EN] Library for numerical array processing / [KR] 수치 계산 및 배열 처리를 위한 라이브러리
import sys                       # [EN] System-specific parameters and functions / [KR] 시스템 관련 기능 (종료 등)

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
# [Type: dict] Chemical composition of the sample (샘플의 화학 조성)
# composition = {'Co':2, 'O':2, 'P':1}
composition = {'C': 30, 'H': 24, 'N': 6, 'Ru': 1}
# [Type: float] Q range settings (Q 최소값, 최대값, 간격)
qmin = 0
qmax = 30
qstep = 0.01

# [EN] Expand composition dictionary into a full list of atom names
# [KR] 조성 딕셔너리를 전체 원자 이름 리스트로 변환 (예: ['Co', 'Co', 'O', 'O', 'P'])
# [Type: list of str]
atom_names = losa.convert_atom_names(composition)

# -----------------------------------------------------------------------------
# Database Loading
# -----------------------------------------------------------------------------
# [EN] Load all atom names and parameters from the database files
# [KR] 데이터베이스 파일에서 모든 원자 이름과 콤프턴 산란 파라미터를 불러옵니다.
# compton_atom_names: [Type: list of str] (DB에 있는 모든 원소 기호)
# compton_scat_parms: [Type: numpy.ndarray] (DB에 있는 모든 파라미터)
compton_atom_names = losa.load_atom_names(compton_aff_element_file)
compton_scat_parms = losa.load_scattering_factors(compton_aff_parm_file)

# [EN] Group atoms to find unique elements and their counts
# [KR] 입력된 샘플의 원자들을 종류별로 그룹화하고 개수를 셉니다.
# atom_unique_names: [Type: list] Unique elements (e.g., ['Co', 'O', 'P'])
# atom_counts: [Type: numpy.ndarray] Counts per element (e.g., [2, 2, 1])
# atom_indices: [Type: numpy.ndarray] Mapping indices (인덱스 매핑)
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
# [EN] Calculate the Compton scattering intensity
# [KR] 콤프턴 산란 강도를 계산합니다.
# list_q: [Type: numpy.ndarray] Q values (X-axis)
# list_compton_scat: [Type: list or numpy.ndarray] Compton intensity (Y-axis)
list_q, list_compton_scat = proc.compton_cal_exp(atom_indices, compton_scat_parms, compton_scat_form_factor,
                            atomic_number, qmin=qmin, qmax=qmax, qstep=qstep, wavelength=wavelength, alpha=alpha)    #proc.compton_calc_exp(XXXX)

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