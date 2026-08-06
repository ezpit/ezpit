import numpy as np
from scipy.spatial.distance import cdist
from scipy import interpolate
import io
# --- [EN] Added imports for Whittaker smoothing / [KR] Whittaker 스무딩 기능을 위해 추가된 라이브러리 ---
import scipy.sparse as sp  # [EN] Sparse matrix package for efficient memory usage / [KR] 메모리를 효율적으로 쓰는 희소 행렬 패키지
from scipy.linalg import cho_factor, cho_solve, LinAlgError  # [EN] Linear algebra solvers (Cholesky decomposition) / [KR] 선형대수 계산 (촐레스키 분해 및 해 구하기)
from math import factorial  # [EN] Mathematical factorial function / [KR] 수학 팩토리얼 함수 (! 연산)
from collections import Counter  # [EN] Dict subclass for counting hashable objects / [KR] 리스트 내 요소의 개수를 세는 도구


def load_atom_name_positions(file_path):
    """
    [EN] Load atom names and their (x, y, z) positions from a file.
    [KR] 파일에서 원자 이름과 (x, y, z) 좌표를 불러옵니다.

    Args:
        file_path (str): [EN] Path to the file / [KR] 파일 경로

    Returns:
        atom_names (list): [EN] List of atom names (e.g., ['O', 'H']) / [KR] 원자 이름 리스트
        atom_positions (numpy.ndarray): [EN] Array of coordinates (N rows, 3 columns) / [KR] (N, 3) 크기의 좌표 배열
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    atom_names = []
    atom_positions = []
    for line in lines:
        parts = line.split()
        atom_names.append(parts[0])
        # [EN] Convert string coordinates to float / [KR] 문자열 좌표를 실수형으로 변환
        atom_positions.append([float(x) for x in parts[1:]])
    atom_positions = np.array(atom_positions)
    return atom_names, atom_positions


def load_atom_names(file_path):
    """
    [EN] Load only a list of atom names from a file.
    [KR] 파일에서 원자 이름 목록만 불러옵니다.

    Returns:
        atom_list (list): [EN] List of strings / [KR] 문자열 리스트
    """
    with open(file_path, 'r') as f:
        atom_list = [line.strip() for line in f.readlines() if line.strip() != '']
    return atom_list


def load_scattering_factors(file_path):
    """
    [EN] Load scattering factor parameters (a, b, c...) from a file.
    [KR] 파일에서 산란 인자 파라미터(a, b, c 등)를 불러옵니다.

    Returns:
        numpy.ndarray: [EN] 2D array of floats / [KR] 실수형 2차원 배열
    """
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            data.append([float(val) for val in line.strip().split('\t')])
    return np.array(data)


def convert_atom_names(composition):
    """
    [EN] Convert composition dict to a full list of atoms.
    [KR] 조성 딕셔너리를 전체 원자 리스트로 변환합니다.
    Ex: {'Co': 2, 'O': 1} -> ['Co', 'Co', 'O']

    Args:
        composition (dict): [EN] Dictionary of element counts / [KR] 원소 개수 사전

    Returns:
        atom_names (list): [EN] Full list of atom symbols / [KR] 전체 원자 기호 리스트
    """
    atom_names = []
    for item in composition.items():
        key, value = item
        for j in range(value):
            atom_names.append(key)
    return atom_names


def get_scattering_factors(atom_names, database_atom_names, database_scat_factors):
    """
    [EN] Retrieve scattering factors for specific atoms from the database.
    [KR] 데이터베이스에서 특정 원자들에 해당하는 산란 인자를 가져옵니다.

    Returns:
        numpy.ndarray: [EN] Array of scattering parameters / [KR] 산란 파라미터 배열
    """
    scat_factors = []
    for atom in atom_names:
        if atom in database_atom_names:
            idx = database_atom_names.index(atom)
            scat_factors.append(database_scat_factors[idx])
        else:
            raise ValueError("There is no atom {0} in database".format(atom))
    return np.asarray(scat_factors)


def get_compton_scattering_factors(atom_names, database_atom_names, database_scat_factors):
    """
    [EN] Retrieve Compton scattering factors and atomic numbers.
    [KR] 콤프턴 산란 인자와 원자 번호를 가져옵니다.

    Returns:
        scat_factors (numpy.ndarray): [EN] Compton parameters / [KR] 콤프턴 파라미터
        atomic_number (list): [EN] List of atomic numbers (integers) / [KR] 원자 번호 리스트 (정수)
    """
    scat_factors = []
    atomic_number = []
    for atom in atom_names:
        if atom in database_atom_names:
            idx = database_atom_names.index(atom)
            scat_factors.append(database_scat_factors[idx])
            atomic_number.append(idx + 1)
        else:
            raise ValueError("There is no atom {0} in database".format(atom))
    return np.asarray(scat_factors), atomic_number


def group_atoms(atom_names):
    """
    [EN] Group identical atoms and count their occurrences.
    [KR] 동일한 원자들을 그룹화하고 그 개수를 셉니다.

    Returns:
        atom_uni_names (list): [EN] Unique atom names / [KR] 고유한 원자 이름
        atom_counts (numpy.ndarray): [EN] Counts per atom / [KR] 원자별 개수
        atom_indices (numpy.ndarray): [EN] Indices mapping to unique names / [KR] 고유 이름에 매핑되는 인덱스
    """
    counter = Counter(atom_names)
    atom_uni_names = list(counter.keys())
    atom_counts = list(counter.values())
    atom_indices = [atom_uni_names.index(atom) for atom in atom_names]
    return atom_uni_names, np.asarray(atom_counts), np.asarray(atom_indices)


def make_folder(file_path):
    """
    [EN] Create a directory if it does not exist.
    [KR] 폴더가 존재하지 않으면 새로 생성합니다.
    """
    file_base = os.path.dirname(file_path)
    if not os.path.exists(file_base):
        try:
            os.makedirs(file_base, exist_ok=True)
        except OSError:
            raise ValueError("Can't create the folder: {}".format(file_base))


def save_txt(filename, q_Iq):
    """
    [EN] Save numpy array data to a text file.
    [KR] Numpy 배열 데이터를 텍스트 파일로 저장합니다.
    """
    make_folder(filename)
    np.savetxt(filename, q_Iq)


def create_atom_distance_matrix(atom_positions):
    """
    [EN] Calculate the pairwise Euclidean distance matrix for all atoms.
         This matrix is essential for the Debye scattering equation (sin(qr)/qr).
    [KR] 모든 원자 쌍 사이의 유클리드 거리 행렬을 계산합니다.
         이 행렬은 Debye 산란 공식 (sin(qr)/qr) 계산에 필수적입니다.

    Args:
        atom_positions (numpy.ndarray):
            [EN] Input array of atomic coordinates with shape (N, 3).
                 N is the number of atoms, and 3 represents (x, y, z) coordinates.
                 (Data Type: float)
            [KR] (N, 3) 크기의 원자 좌표 입력 배열입니다.
                 N은 원자의 개수이며, 3은 (x, y, z) 좌표를 나타냅니다.
                 (데이터 타입: float)

    Returns:
        distance_matrix (numpy.ndarray):
            [EN] Output symmetric distance matrix with shape (N, N).
                 Element (i, j) represents the Euclidean distance between atom i and atom j.
                 (Data Type: float)
            [KR] (N, N) 크기의 대칭 거리 행렬 결과입니다.
                 (i, j) 요소는 i번째 원자와 j번째 원자 사이의 유클리드 거리를 나타냅니다.
                 (데이터 타입: float)
    """
    # [EN] Calculate distance between every pair of points using scipy.spatial.distance.cdist
    # [KR] scipy.spatial.distance.cdist를 사용하여 모든 점 쌍 사이의 거리를 계산합니다.
    distance_matrix = cdist(atom_positions, atom_positions)

    return distance_matrix

def __cal_fi(scat_values, q):
    """
    [EN] Calculate atomic form factor f(q) using 5-Gaussian approximation.
    [KR] 5-가우시안 근사를 사용하여 원자 형상 인자 f(q)를 계산합니다.
    Formula: f(q) = sum(ai * exp(-bi * (q/4pi)^2)) + c

    Args:
        scat_values (list): [EN] Coefficients [a1, b1, ..., c] / [KR] 계수 리스트
        q (float/array): [EN] Momentum transfer / [KR] 운동량 변화량 Q

    Returns:
        fi (float/array): [EN] Calculated form factor / [KR] 계산된 형상 인자
    """
    k_sq = (0.25 * q / np.pi) ** 2
    fi1 = scat_values[0] * np.exp(-scat_values[1] * k_sq)
    fi2 = scat_values[2] * np.exp(-scat_values[3] * k_sq)
    fi3 = scat_values[4] * np.exp(-scat_values[5] * k_sq)
    fi4 = scat_values[6] * np.exp(-scat_values[7] * k_sq)
    fi5 = scat_values[8] * np.exp(-scat_values[9] * k_sq)
    fic = scat_values[10]
    fi = fi1 + fi2 + fi3 + fi4 + fi5 + fic
    return fi


def __cal_compton_fi(compton_scattering_factors, q):
    """
    [EN] Calculate Compton scattering form factor (Inelastic scattering).
    [KR] 콤프턴 산란 형상 인자를 계산합니다 (비탄성 산란).
    """
    k_sq = (0.25 * q / np.pi) ** 2
    fi1 = compton_scattering_factors[0] * np.exp(-compton_scattering_factors[6] * k_sq)
    fi2 = compton_scattering_factors[1] * np.exp(-compton_scattering_factors[7] * k_sq)
    fi3 = compton_scattering_factors[2] * np.exp(-compton_scattering_factors[8] * k_sq)
    fi4 = compton_scattering_factors[3] * np.exp(-compton_scattering_factors[9] * k_sq)
    fi5 = compton_scattering_factors[4] * np.exp(-compton_scattering_factors[10] * k_sq)
    fic = compton_scattering_factors[5]
    fi = fi1 + fi2 + fi3 + fi4 + fi5 + fic
    return fi


def compton_cal_exp(atom_indices, compton_scat_parms, compton_scattering_factors,
                    atomic_number, qmin, qmax, qstep, wavelength, alpha):

    # ============================================================================
    #  Compton (incoherent) X-ray scattering calculation
    #  컴프턴(비탄성) X선 산란 강도 계산
    # ----------------------------------------------------------------------------
    #  REFERENCE / 참고문헌
    #  H. H. M. Balyuzi, "Analytic Approximations to Incoherently Scattered
    #  X-Ray Intensities", Acta Cryst. (1975). A31, 600-602.
    #
    #  Source data of the tabulated coefficients (compton_parameter_only.txt):
    #  계수 테이블의 원 출처:
    #    DABAX file "CrossSec_Compton_Balyuzi.dat", extracted from sf_inc.f of
    #    the library by S. Brennan and P. L. Cowan, Rev. Sci. Instrum. 63, 850
    #    (1992). Coefficients originally from Cromer & Mann, J. Chem. Phys. 47,
    #    1892 (1967) and Cromer, J. Chem. Phys. 50, 4857 (1969), fitted by
    #    Balyuzi (1975).
    #
    #  KEY IDEA / 핵심 개념
    #  --------------------------------------------------------------------------
    #  EN: Balyuzi did NOT fit the incoherent intensity I_inc(s) directly.
    #      He fitted the function  F(s) = Z - I_inc(s)  to a sum of five
    #      Gaussians (no constant term, c = 0):
    #
    #          F_fit(s) = c + sum_{i=1..5} a_i * exp(-b_i * s^2),   c = 0
    #
    #      where  s = sin(theta)/lambda = Q / (4*pi),  and  Z = atomic number.
    #      Therefore the incoherent (Compton) intensity per atom is recovered by
    #      a simple subtraction:
    #
    #          I_inc(s) = Z - F_fit(s)
    #
    #      F_fit is ALREADY (Z - I_inc); it is NOT an atomic form factor, so it
    #      must not be squared or divided by Z. (This was the previous bug.)
    #
    #  KR: Balyuzi는 비탄성 강도 I_inc(s)를 직접 피팅한 것이 아니라,
    #      함수  F(s) = Z - I_inc(s)  를 5개의 가우시안 합(상수항 c = 0)으로
    #      피팅했다:
    #
    #          F_fit(s) = c + Σ_{i=1..5} a_i · exp(-b_i · s^2),   c = 0
    #
    #      여기서  s = sin(theta)/lambda = Q / (4π),  Z = 원자번호 이다.
    #      따라서 원자당 비탄성(컴프턴) 강도는 단순한 뺄셈으로 얻는다:
    #
    #          I_inc(s) = Z - F_fit(s)
    #
    #      F_fit 은 그 자체가 이미 (Z - I_inc) 값이며, 원자 form factor가
    #      아니다. 그러므로 제곱하거나 Z로 나누면 안 된다. (이전 버그의 원인)
    #
    #  VALIDITY RANGE / 유효 범위
    #  --------------------------------------------------------------------------
    #  EN: The fit is accurate for s = sin(theta)/lambda <= ~1.4-1.5 A^-1,
    #      i.e. Q <= ~18-19 A^-1. Beyond this, values are extrapolated.
    #  KR: 이 피팅은 s = sin(theta)/lambda <= 약 1.4~1.5 A^-1 (즉 Q <= 약
    #      18~19 A^-1) 범위에서 정확하다. 그 이상은 외삽 영역이다.
    # ============================================================================

    """
    [EN] Calculate total experimental Compton scattering intensity.
         Includes Breit-Dirac recoil factor correction.
    [KR] 전체 실험적 콤프턴 산란 강도를 계산합니다.
         Breit-Dirac 반동(Recoil) 보정이 포함됩니다.

    Args:
        wavelength (float): [EN] X-ray wavelength (Angstrom) / [KR] X선 파장
        alpha (float): [EN] Recoil parameter (usually 2 or 3) / [KR] 반동 파라미터

    Returns:
        q_range (numpy.ndarray): [EN] Q axis / [KR] Q 축 데이터
        list_compton_scat (list): [EN] Compton intensity / [KR] 콤프턴 산란 강도
    """
    me = 9.109534e-31  # [EN] Electron mass (kg) / [KR] 전자 질량
    C = 2.99792458e18  # [EN] Speed of light (Angstrom/s) / [KR] 빛의 속도
    h = 6.62607015e-14  # [EN] Planck constant / [KR] 플랑크 상수
    part_A = 2.0 * h * wavelength / me / C

    num_atom = len(atom_indices)
    num_fact = len(compton_scattering_factors)
    q_range = np.arange(qmin, qmax, qstep)
    list_compton_scat = []

    for q in q_range:
        atomic_number_sum = 0.0
        ffit_sum = 0.0
        #fi2_sum = 0.0
        part_B = (q / (4.0 * np.pi)) ** 2.0
        # [EN] Breit-Dirac recoil factor calculation
        # [KR] Breit-Dirac 반동 인자 계산
        BD_recoil_fact = (part_A * part_B + 1) ** (-alpha)
        list_fi = []
        for k in range(num_fact):
            list_fi.append(__cal_compton_fi(compton_scat_parms[atomic_number[k] - 1], q))
        list_fi = np.asarray(list_fi)
        for i, idx in enumerate(atom_indices):
            fi = list_fi[idx]
            atomic_number_sum = atomic_number_sum + atomic_number[idx]
            ffit_sum = ffit_sum + fi  # I_inc = Z - F_fit (Balyuzi 1975); F_fit is already (Z - I_inc)
            # fi2_sum = fi2_sum + fi   #removed due to a wwrong equation (fi ** 2) / atomic_number[idx]
        compton_scat = BD_recoil_fact * (1 / num_atom * atomic_number_sum - 1 / num_atom * ffit_sum)
        #compton_scat = BD_recoil_fact * (1 / num_atom * atomic_number_sum - 1 / num_atom * fi2_sum)
        list_compton_scat.append(compton_scat)
    return q_range, list_compton_scat


# ----------------------------------------------------------------------------------
# Lorch Function Added (Requested 12/28/2025)
# ----------------------------------------------------------------------------------
def lorch_function_sinc(Q, Q_max):
    """
    [EN] Calculate the Lorch modification function M(Q).
         Used to reduce termination ripples in FFT.
    [KR] Lorch 수정 함수 M(Q)를 계산합니다.
         FFT 시 발생하는 종료 리플(Termination Ripple, 물결무늬 노이즈)을 줄이는 데 사용됩니다.
    Formula: M(Q) = sin(pi * Q / Q_max) / (pi * Q / Q_max)
    """
    Q = np.asarray(Q)
    if Q_max == 0:
        return np.ones_like(Q)
    # [EN] numpy.sinc(x) is mathematically sin(pi*x)/(pi*x)
    # [KR] numpy.sinc(x) 함수는 수학적으로 sin(pi*x)/(pi*x) 입니다.
    return np.sinc(Q / Q_max)


def apply_lorch_function(Q, FQ):
    """
    [EN] Apply the Lorch function to F(Q) data.
    [KR] F(Q) 데이터에 Lorch 함수를 적용(곱하기)합니다.

    Args:
        Q (numpy.ndarray): [EN] Q values / [KR] Q 값 배열
        FQ (numpy.ndarray): [EN] F(Q) intensity values / [KR] F(Q) 강도 배열

    Returns:
        modified_FQ (numpy.ndarray): [EN] Lorch-modified F(Q) / [KR] Lorch 보정된 F(Q)
    """
    Q = np.asarray(Q)
    FQ = np.asarray(FQ)
    Q_max = np.max(Q)
    lorch_values = lorch_function_sinc(Q, Q_max)
    return FQ * lorch_values


def cal_Iq(atom_indices, scattering_factors, atom_distance_matrix,
           qmin=0.5, qmax=20, qstep=0.05):
    """
    [EN] Calculate the total Scattering Intensity I(q) based on the Debye scattering equation.
         This function computes the sum of coherent scattering from all atom pairs.
         Formula: I(q) = sum_i sum_j [ f_i(q) * f_j(q) * sin(q*r_ij)/(q*r_ij) ]

    [KR] Debye 산란 공식을 기반으로 총 산란 강도 I(q)를 계산합니다.
         모든 원자 쌍 사이의 간섭성 산란 합을 구합니다.
         공식: I(q) = sum_i sum_j [ f_i(q) * f_j(q) * sin(q*r_ij)/(q*r_ij) ]

    Args:
        atom_indices (list or numpy.ndarray):
            [EN] A 1D array of integers (length N) mapping each atom to its element type index.
            [KR] 각 원자가 어떤 원소 종류인지 매핑하는 정수형 1차원 배열 (길이 N).

        scattering_factors (numpy.ndarray):
            [EN] A 2D array of atomic scattering factor coefficients (shape: [num_elements, 11]).
                 Used to calculate f(q).
            [KR] 원자 산란 인자 계수를 담은 2차원 배열. f(q) 계산에 사용됩니다.

        atom_distance_matrix (numpy.ndarray):
            [EN] A symmetric matrix (N x N) representing the Euclidean distance between atom i and atom j.
            [KR] i번째 원자와 j번째 원자 사이의 유클리드 거리를 나타내는 대칭 행렬 (N x N).

        qmin (float, optional):
            [EN] Minimum Q value (inverse Angstrom). Default is 0.5.
            [KR] Q 값의 최소값 (1/A). 기본값은 0.5.

        qmax (float, optional):
            [EN] Maximum Q value (inverse Angstrom). Default is 20.
            [KR] Q 값의 최대값 (1/A). 기본값은 20.

        qstep (float, optional):
            [EN] Step size for Q. Default is 0.05.
            [KR] Q 계산 간격. 기본값은 0.05.

    Returns:
        tuple (numpy.ndarray, list):
            [EN] Returns a tuple containing:
                 - q_range: Array of Q values.
                 - list_Iq: List of calculated Intensity values I(q).
            [KR] 다음을 포함하는 튜플을 반환합니다:
                 - q_range: Q 값 배열.
                 - list_Iq: 계산된 산란 강도 I(q) 리스트.
    """
    # Theoretical calculation loop (kept as loop to prevent memory overflow for large N)
    num_atom = len(atom_indices)
    num_fact = len(scattering_factors)
    diag_idx = np.diag_indices(num_atom)
    distance_matrix_non_zero = np.copy(atom_distance_matrix)
    distance_matrix_non_zero[diag_idx] = 1.0
    list_Iq = []
    q_range = np.arange(qmin, qmax, qstep)
    for q in q_range:
        fi_mat = np.zeros((num_atom, num_atom))
        list_fi = []
        for k in range(num_fact):
            list_fi.append(__cal_fi(scattering_factors[k], q))
        list_fi = np.asarray(list_fi)
        for i, idx in enumerate(atom_indices):
            fi = list_fi[idx]
            fi_mat[i, :] = fi
        if q == 0:
            sin_mat = np.ones(np.shape(atom_distance_matrix))
        else:
            sin_mat = np.sin(q * atom_distance_matrix) / (
                    q * distance_matrix_non_zero)
        sin_mat[diag_idx] = 1.0
        Iq = fi_mat * np.transpose(fi_mat) * sin_mat
        list_Iq.append(np.sum(Iq))
    return q_range, list_Iq


def cal_Sq(atom_indices, scattering_factors, atom_distance_matrix,
           qmin=0.5, qmax=20, qstep=0.05, return_Iq=False):
    """
    [EN] Calculate theoretical Structure Factor S(q) using the Debye scattering equation.
         This function computes I(q), S(q), and F(q) from atomic coordinates and scattering factors.
    [KR] Debye 산란 공식을 사용하여 이론적인 구조 인자 S(q)를 계산합니다.
         원자 좌표와 산란 인자를 바탕으로 I(q), S(q), F(q)를 계산합니다.

    Args:
        atom_indices (list or numpy.ndarray):
            [EN] A 1D array of integers mapping each atom to its element type index.
                 (e.g., [0, 0, 1] means Atom1=Type0, Atom2=Type0, Atom3=Type1)
            [KR] 각 원자가 어떤 원소 종류인지 나타내는 정수형 인덱스 배열입니다.
                 (예: [0, 0, 1]은 원자1=타입0, 원자2=타입0, 원자3=타입1을 의미)

        scattering_factors (numpy.ndarray):
            [EN] A 2D array of scattering coefficients (shape: [num_unique_elements, 11]).
                 Contains Gaussian parameters (a, b, c) for f(q) calculation.
            [KR] 산란 계수들을 포함하는 2차원 배열입니다 (크기: [고유원소수, 11]).
                 f(q) 계산을 위한 가우시안 파라미터(a, b, c)를 포함합니다.

        atom_distance_matrix (numpy.ndarray):
            [EN] A symmetric matrix (N x N) of pairwise Euclidean distances between atoms.
                 Calculated by 'create_atom_distance_matrix'.
            [KR] 원자들 사이의 쌍(pairwise) 유클리드 거리를 담은 대칭 행렬 (N x N)입니다.
                 'create_atom_distance_matrix' 함수로 계산된 값입니다.

        qmin (float):
            [EN] Minimum Q value (inverse Angstrom). Default is 0.5.
            [KR] Q 값의 최소범위 (1/A). 기본값은 0.5입니다.

        qmax (float):
            [EN] Maximum Q value (inverse Angstrom). Default is 20.
            [KR] Q 값의 최대범위 (1/A). 기본값은 20입니다.

        qstep (float):
            [EN] Step size for Q iteration. Default is 0.05.
            [KR] Q 계산 간격. 기본값은 0.05입니다.

        return_Iq (bool):
            [EN] If True, returns detailed data including I(q), F(q), and form factor averages.
            [KR] True일 경우, I(q), F(q), 형상 인자 평균값 등 상세 데이터를 함께 반환합니다.

    Returns:
        If return_Iq is False:
            (q_range, list_Sq)
            - q_range (numpy.ndarray): [EN] Generated Q axis / [KR] 생성된 Q 축 데이터
            - list_Sq (numpy.ndarray): [EN] Calculated Structure Factor S(q) / [KR] 계산된 구조 인자 S(q)

        If return_Iq is True:
            (q_range, list_Iq, list_Sq, list_Fq, mean_sq_fi, sq_mean_fi)
            - list_Iq (numpy.ndarray): [EN] Total Scattering Intensity I(q) / [KR] 총 산란 강도 I(q)
            - list_Fq (numpy.ndarray): [EN] Reduced Structure Function F(q) = Q * (S(q) - 1) / [KR] 환원 구조 함수
            - mean_sq_fi (numpy.ndarray): [EN] Average of squared form factors <f^2> / [KR] 형상 인자 제곱의 평균
            - sq_mean_fi (numpy.ndarray): [EN] Squared average of form factors <f>^2 / [KR] 평균 형상 인자의 제곱
    """
    # Theoretical calculation loop
    num_atom = len(atom_indices)
    num_fact = len(scattering_factors)
    diag_idx = np.diag_indices(num_atom)
    distance_matrix_non_zero = np.copy(atom_distance_matrix)
    distance_matrix_non_zero[diag_idx] = 1.0

    list_Iq = []
    sq_mean_fi = []
    mean_sq_fi = []
    q_range = np.arange(qmin, qmax, qstep)

    for q in q_range:
        fi_mat = np.zeros((num_atom, num_atom))
        list_fi = []
        for k in range(num_fact):
            list_fi.append(__cal_fi(scattering_factors[k], q))
        list_fi = np.asarray(list_fi)
        fi_sum, fi2_sum = 0.0, 0.0
        for i, idx in enumerate(atom_indices):
            fi = list_fi[idx]
            fi_mat[i, :] = fi
            fi_sum = fi_sum + fi
            fi2_sum = fi2_sum + fi ** 2
        if q == 0:
            sin_mat = np.ones(np.shape(atom_distance_matrix))
        else:
            sin_mat = np.sin(q * atom_distance_matrix) / (
                    q * distance_matrix_non_zero)
        sin_mat[diag_idx] = 1.0
        Iq = fi_mat * np.transpose(fi_mat) * sin_mat
        list_Iq.append(np.sum(Iq))
        sq_mean_fi.append((fi_sum / num_atom) ** 2)
        mean_sq_fi.append(fi2_sum / num_atom)

    list_Iq = np.asarray(list_Iq)
    mean_sq_fi = np.asarray(mean_sq_fi)  # <f^2>
    sq_mean_fi = np.asarray(sq_mean_fi)  # <f>^2
    list_Sq = (list_Iq - num_atom * mean_sq_fi) / (num_atom * sq_mean_fi) + 1
    list_Fq = q_range * (list_Sq - 1)

    if return_Iq:
        return q_range, list_Iq, list_Sq, list_Fq, mean_sq_fi, sq_mean_fi
    else:
        return q_range, list_Sq


def cal_Gr_integral(q, Sq, rmin=0, rmax=100, rstep=0.02, qdamp=0.0):
    """
    [EN] Calculate the Pair Distribution Function G(r) from Structure Factor S(q) using a direct sine integral transform.
         This method performs a discrete integration (Riemann sum) for each r value.
         It is generally slower than FFT but free from grid artifacts and padding issues.
         Formula: G(r) = (2/pi) * integral [ Q * (S(Q)-1) * sin(Q*r) ] dQ

    [KR] 구조 인자 S(q)로부터 직접 사인 적분 변환을 사용하여 쌍 분포 함수 G(r)을 계산합니다.
         각 r 값에 대해 이산 적분(리만 합)을 수행합니다.
         FFT 방식보다 속도는 느리지만 그리드 아티팩트나 패딩 문제 없이 정확한 값을 얻을 수 있습니다.
         공식: G(r) = (2/pi) * integral [ Q * (S(Q)-1) * sin(Q*r) ] dQ

    Args:
        q (numpy.ndarray):
            [EN] The momentum transfer axis Q (inverse Angstrom). 1D array of floats.
            [KR] 운동량 전달 축 Q (단위: 1/A). 실수형 1차원 배열.

        Sq (numpy.ndarray):
            [EN] The Structure Factor S(q) corresponding to the q axis. 1D array of floats.
            [KR] Q 축에 해당하는 구조 인자 S(q). 실수형 1차원 배열.

        rmin (float, optional):
            [EN] Minimum value for the real-space distance r (Angstrom). Default is 0.
            [KR] 실공간 거리 r의 최소값 (단위: A). 기본값은 0.

        rmax (float, optional):
            [EN] Maximum value for the real-space distance r (Angstrom). Default is 100.
            [KR] 실공간 거리 r의 최대값 (단위: A). 기본값은 100.

        rstep (float, optional):
            [EN] Step size for the r axis. Default is 0.02.
            [KR] r 축의 간격. 기본값은 0.02.

        qdamp (float, optional):
            [EN] Gaussian damping factor to simulate limited instrumental resolution.
                 Applied as envelope: exp(-0.5 * (r * qdamp)^2). Default is 0.0 (no damping).
            [KR] 제한된 기기 해상도를 시뮬레이션하기 위한 가우시안 감쇠 인자.
                 적용 식: exp(-0.5 * (r * qdamp)^2). 기본값은 0.0 (감쇠 없음).

    Returns:
        tuple (numpy.ndarray, numpy.ndarray):
            [EN] Returns a tuple containing:
                 - list_r: The generated r-axis (Angstrom).
                 - list_Gr: The calculated G(r).
            [KR] 다음을 포함하는 튜플을 반환합니다:
                 - list_r: 생성된 r 축 (단위: A).
                 - list_Gr: 계산된 G(r) 값.
    """
    list_r = np.arange(rmin, rmax + rstep, rstep)
    Fq = (Sq - 1) * q
    qstep = q[1] - q[0]
    list_Gr = np.zeros_like(list_r)
    list_qFq = Fq * qstep
    for i, r in enumerate(list_r):
        list_Gr[i] = np.sum(list_qFq * np.sin(q * r)) * 2.0 / np.pi
    if qdamp != 0.0:
        list_Gr = np.exp(-0.5 * (list_r * qdamp) ** 2) * list_Gr
    return list_r, list_Gr


def cal_Gr_fft(q, Sq, rmin=0, rmax=100, rstep=0.02, qdamp=0.0,
                     extrapolate_type="linear"):
    """
    [EN] Calculate the Pair Distribution Function G(r) from Structure Factor S(q) using Inverse Fast Fourier Transform (IFFT).
         This function handles low-Q extrapolation and zero-padding to achieve the desired real-space resolution (rstep).
         Formula: G(r) = (2/pi) * FFT_imag[ Q * (S(Q)-1) ]

    [KR] 고속 푸리에 변환(IFFT)을 사용하여 구조 인자 S(q)로부터 쌍 분포 함수 G(r)을 계산합니다.
         Low-Q 영역의 보간(extrapolation)과 제로 패딩(zero-padding)을 처리하여 원하는 실공간 해상도(rstep)를 얻습니다.
         공식: G(r) = (2/pi) * FFT_imag[ Q * (S(Q)-1) ]

    Args:
        q (numpy.ndarray):
            [EN] The momentum transfer axis Q (inverse Angstrom). 1D array of floats.
            [KR] 운동량 전달 축 Q (단위: 1/A). 실수형 1차원 배열.

        Sq (numpy.ndarray):
            [EN] The Structure Factor S(q) corresponding to the q axis.
            [KR] Q 축에 해당하는 구조 인자 S(q).

        rmin (float, optional):
            [EN] Minimum value for the real-space distance r (Angstrom). Default is 0.
            [KR] 실공간 거리 r의 최소값 (단위: A). 기본값은 0.

        rmax (float, optional):
            [EN] Maximum value for the real-space distance r (Angstrom). Default is 100.
            [KR] 실공간 거리 r의 최대값 (단위: A). 기본값은 100.

        rstep (float, optional):
            [EN] Step size for the r axis. Smaller steps require larger zero-padding in Q-space. Default is 0.02.
            [KR] r 축의 간격. 간격이 작을수록 Q 공간에서 더 많은 제로 패딩이 필요합니다. 기본값은 0.02.

        qdamp (float, optional):
            [EN] Gaussian damping factor for instrument resolution correction.
                 Applied in real space as: G(r) * exp(-0.5 * (r * qdamp)^2). Default is 0.0.
            [KR] 기기 해상도 보정을 위한 가우시안 감쇠 인자.
                 실공간에서 G(r) * exp(-0.5 * (r * qdamp)^2) 형태로 적용됩니다. 기본값은 0.0.

        extrapolate_type (str, optional):
            [EN] Interpolation method for filling the low-Q region (0 to Qmin).
                 Options: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'. Default is 'linear'.
            [KR] Low-Q 영역 (0부터 Qmin까지)을 채우기 위한 보간 방법.
                 옵션: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'. 기본값은 'linear'.

    Returns:
        tuple (numpy.ndarray, numpy.ndarray):
            [EN] Returns a tuple containing:
                 - r_list: The generated r-axis (Angstrom).
                 - gr: The calculated G(r) interpolated onto r_list.
            [KR] 다음을 포함하는 튜플을 반환합니다:
                 - r_list: 생성된 r 축 (단위: A).
                 - gr: r_list에 맞춰 보간된 G(r) 값.
    """
    qstep = q[1] - q[0]
    num_point = int(np.ceil(q[0] / qstep))
    Fq = (Sq - 1) * q
    pad_Fq = np.zeros(num_point)
    if q[0] > 0.0:
        f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate",
                                       kind=extrapolate_type)
        pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep)
    Fq_vals = np.append(pad_Fq, Fq)
    r_list = np.arange(rmin, rmax + rstep, rstep)

    # ------------------------------------------------------------------
    # Nyquist check (same as cal_expGr_fft):
    #   The IFFT produces total_point values; only the first half
    #   (indices 0 .. total_point//2) corresponds to physically valid
    #   positive-r values, giving r_nyquist = pi / qstep.
    #   If rmax exceeds this limit, interpolate F(q) onto a finer q grid.
    # ------------------------------------------------------------------
    r_nyquist = np.pi / qstep
    if rmax > r_nyquist:
        qstep_needed = np.pi / rmax
        q_orig = np.arange(len(Fq_vals)) * qstep
        q_fine = np.arange(q_orig[0], q_orig[-1] + qstep_needed, qstep_needed)
        Fq_vals = np.interp(q_fine, q_orig, Fq_vals)
        qstep = qstep_needed

    num_point = len(Fq_vals)
    total_point = int(2 * np.pi / (rstep * qstep))
    Fq_pad = np.pad(Fq_vals, (0, total_point - num_point), mode="constant")
    norm = total_point * qstep * 2 / np.pi
    gr_full = norm * np.imag(np.fft.ifft(Fq_pad))

    # Use only the alias-free first half of the IFFT output
    half = total_point // 2
    rfine = np.arange(half) * rstep
    gr_fine = gr_full[:half]
    if qdamp != 0.0:
        gr_fine = gr_fine * np.exp(-0.5 * (rfine * qdamp) ** 2)
    gr = np.interp(r_list, rfine, gr_fine)
    return r_list, gr


# ----------------------------------------------------------------------------------
# [Cal_expSq] Core Analysis Function / 핵심 분석 함수
# ----------------------------------------------------------------------------------
def cal_expSq(atom_indices, scattering_factors, expqiq, bkgqiq, qmin=0, qmax=25, qstep=0.01,
              background_scale=1.1, poly_order=11.0, return_Iq=False):
    """
    [EN] Core function to calculate Structure Factor S(q) and F(q) from experimental I(q).
    [KR] 실험 데이터 I(q)로부터 구조 인자 S(q)와 F(q)를 계산하는 핵심 함수입니다.

    Steps (단계):
    1. Load Data (데이터 로드)
    2. Interpolate (보간: 데이터를 동일한 간격으로 맞춤)
    3. Background Subtraction (배경 제거)
    4. Calculate Theoretical Form Factors (이론적 형상 인자 계산)
    5. Calculate S(q) (S(q) 계산)
    6. Polynomial Correction (다항식 보정)

    Args:
        poly_order (float): [EN] Order of polynomial for correction / [KR] 보정용 다항식 차수
        background_scale (float): [EN] Scaling factor for background / [KR] 배경 제거 스케일 팩터

    Returns:
        Tuple containing q, Iq, S(q), F(q), etc. / Q, I(q), S(q), F(q) 등을 포함한 튜플
    """
    # 1. Load Data
    if isinstance(expqiq, str):
        # [EN] If input is a file path string, load from file
        # [KR] 입력이 파일 경로 문자열이면 파일에서 로드
        data_exp = load_qiq_file(expqiq, min_cols=2, usecols=(0, 1))
        exp_q = data_exp[:, 0]
        exp_Iq = data_exp[:, 1]
    else:
        # [EN] If input is already an array
        # [KR] 입력이 이미 배열인 경우
        exp_q = expqiq[0]
        exp_Iq = expqiq[1]

    if isinstance(bkgqiq, str):
        data_bkg = load_qiq_file(bkgqiq, min_cols=2, usecols=(0, 1))
        bkg_Iq = data_bkg[:, 1]
    elif bkgqiq is not None:
        if hasattr(bkgqiq, 'shape') and len(bkgqiq.shape) > 1:
            bkg_Iq = bkgqiq[1]
        elif isinstance(bkgqiq, (list, tuple)) and len(bkgqiq) == 2 and hasattr(bkgqiq[0], '__len__'):
            bkg_Iq = bkgqiq[1]
        else:
            bkg_Iq = bkgqiq
    else:
        bkg_Iq = np.zeros_like(exp_Iq)  # [EN] Fill with zeros if no background / [KR] 배경 데이터가 없으면 0으로 채움

    # 2. Interpolate
    q_range = np.arange(qmin, qmax, qstep)
    scaled_expIq = np.interp(q_range, exp_q, exp_Iq)  # [EN] Interpolate exp data / [KR] 실험 데이터 보간
    scaled_bkgIq = np.interp(q_range, exp_q, bkg_Iq)  # [EN] Interpolate bkg data / [KR] 배경 데이터 보간

    # 3. Background Subtraction
    # [EN] Calculate Net Intensity: I_net = I_exp - scale * I_bkg
    # [KR] 순수 강도 계산: 실험값 - (스케일 * 배경값)
    list_scaled_bkgIq = background_scale * scaled_bkgIq
    list_Iq = scaled_expIq - list_scaled_bkgIq

    # 4. Calculate Form Factors (VECTORIZED)
    # [EN] Vectorized form factor calculation — removes 'for q in q_range' loop
    #      for significant speedup. Result is identical to the loop version.
    # [KR] 벡터화된 형상 인자 계산 — 'for q in q_range' 루프 제거로 속도 향상.
    #      결과는 루프 버전과 동일합니다.
    num_atom = len(atom_indices)

    # [EN] (1) Pre-compute form factors for ALL elements over the entire q_range.
    #          __cal_fi supports numpy array input, so we get a 2D array at once.
    #          Result shape: (num_unique_elements, len(q_range))
    # [KR] (1) 모든 unique element에 대해 전체 q_range의 form factor를 한번에 계산.
    #          __cal_fi가 numpy 배열 연산을 지원하므로 2D 배열로 한번에 얻음.
    #          결과 shape: (num_unique_elements, len(q_range))
    list_fi_all_q = np.array([__cal_fi(sf, q_range) for sf in scattering_factors])

    # [EN] (2) Broadcast to each atom via atom_indices.
    #          Result shape: (num_atom, len(q_range))
    # [KR] (2) atom_indices를 이용해 각 원자별 form factor 배열로 확장.
    #          결과 shape: (num_atom, len(q_range))
    atom_fi_all_q = list_fi_all_q[atom_indices]

    # [EN] (3) Sum over atom axis (axis=0) → 1D array of shape (len(q_range),)
    # [KR] (3) Atom 방향(axis=0)으로 합계 → (len(q_range),) 크기 1D 배열
    sum_fi    = np.sum(atom_fi_all_q,       axis=0)   # Σ f_i
    sum_fi_sq = np.sum(atom_fi_all_q ** 2,  axis=0)   # Σ f_i²

    sq_mean_fi = (sum_fi / num_atom) ** 2             # <f>²
    mean_sq_fi = sum_fi_sq / num_atom                 # <f²>

    # 6. Calculate S(q) — PDFgetX3 normalization (default)
    # ------------------------------------------------------------------
    # [EN] X-ray scattering-factor normalization following the ad hoc
    #      data-reduction approach of PDFgetX3. The experimental I(q) is in
    #      arbitrary units (not normalized per incident flux nor per number of
    #      scatterers), so a least-squares scale factor is required to bring the
    #      normalized intensity onto the normal scattering-factor curve before
    #      computing S(q). Without this scale the S(q) amplitude is wrong
    #      (the deviation from 1 is off by a large factor).
    #        normalized_intensity     = I / <f>^2
    #        normal_scattering_factor = <f^2> / <f>^2
    #        normalization_scale = dot(normalized_intensity, normal_scattering_factor)
    #                              / dot(normalized_intensity, normalized_intensity)
    #        S(q) = normalization_scale * normalized_intensity
    #               - normal_scattering_factor + 1
    # [KR] PDFgetX3의 ad hoc 데이터 처리 방식을 따르는 X-ray 산란인자 정규화.
    #      실험 I(q)는 임의 단위(입사 강도/산란체 수로 정규화되지 않음)이므로,
    #      정규화 강도를 normal scattering-factor 곡선에 맞추는 최소자승 스케일
    #      상수가 필요하다. 이 스케일이 없으면 S(q) 진폭이(1로부터의 편차가)
    #      큰 배수로 어긋난다.
    #
    # Reference:
    #   Juhas, P., Davis, T., Farrow, C. L. & Billinge, S. J. L. (2013).
    #   PDFgetX3: a rapid and highly automatable program for processing powder
    #   diffraction data into total scattering pair distribution functions.
    #   J. Appl. Cryst. 46, 560-566.
    # ------------------------------------------------------------------
    normalized_intensity     = list_Iq / sq_mean_fi        # I / <f>^2   (sq_mean_fi = <f>^2)
    normal_scattering_factor = mean_sq_fi / sq_mean_fi     # <f^2> / <f>^2 (mean_sq_fi = <f^2>)
    normalization_scale = (np.dot(normalized_intensity, normal_scattering_factor)
                           / np.dot(normalized_intensity, normalized_intensity))
    list_Sq = (normalization_scale * normalized_intensity
               - normal_scattering_factor + 1)

    # ------------------------------------------------------------------
    # [EN] Standard formula (kept for reference, NOT used by default).
    #      To revert to the standard formula, comment out the PDFgetX3 block
    #      above and uncomment the line below.
    # [KR] 표준 공식 (참고용으로 보존, 기본 사용 안 함).
    #      표준 공식으로 되돌리려면 위 PDFgetX3 블록을 주석 처리하고
    #      아래 줄의 주석을 해제하세요.
    #
    #   list_Sq = (list_Iq - num_atom * mean_sq_fi) / (num_atom * sq_mean_fi) + 1
    # ------------------------------------------------------------------

    # 7. Polynomial Correction for F(q)
    qmax_for_sq = float(np.max(q_range))

    # [EN] Helper function to calculate polynomial fit
    # [KR] 다항식 적합(Fit)을 계산하는 내부 함수
    def _poly_for_sq_at_order(order_int: int):
        order_int = max(int(order_int), 1)
        qscaled = q_range / qmax_for_sq
        V = np.vander(qscaled, order_int + 1)
        A = V[:, :-1]
        b = q_range * (list_Sq - 1.0)
        p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # [EN] Least squares fit / [KR] 최소자승법
        vand2 = np.vander([qmax_for_sq], order_int + 1)[0, :-1]
        new_p = p / vand2
        final_p = np.poly1d(new_p)
        return final_p(q_range)

    # [EN] Handle non-integer polynomial orders
    # [KR] 소수점 차수(Non-integer) 다항식 처리 (두 정수 차수의 가중 평균)
    po = float(poly_order)
    if po.is_integer():
        polynomial_for_sq = _poly_for_sq_at_order(int(round(po)))
    else:
        lo = int(np.floor(po))
        hi = int(np.ceil(po))
        w_hi = po - lo
        w_lo = 1.0 - w_hi
        poly_lo = _poly_for_sq_at_order(lo)
        poly_hi = _poly_for_sq_at_order(hi)
        polynomial_for_sq = w_lo * poly_lo + w_hi * poly_hi

    # [EN] Normalize S(q) and calculate F(q)
    # [KR] S(q) 정규화 및 F(q) 계산
    norm_list_Sq = list_Sq - polynomial_for_sq  # [EN] Remove polynomial background / [KR] 다항식 배경 제거
    list_Fq = q_range * (norm_list_Sq - 1.0)

    if return_Iq:
        return (q_range, list_Iq, scaled_expIq, list_scaled_bkgIq, list_Sq, norm_list_Sq, list_Fq, mean_sq_fi, sq_mean_fi,
                polynomial_for_sq, normalized_intensity,normal_scattering_factor, normalization_scale)
    else:
        return (q_range, list_Iq, scaled_expIq, list_scaled_bkgIq, list_Sq, norm_list_Sq, list_Fq, mean_sq_fi, sq_mean_fi,
                polynomial_for_sq, normalized_intensity,normal_scattering_factor, normalization_scale)


def cal_fq(qmin, qmax, Sq, qstep=0.01):
    """
    [EN] Calculate the Reduced Structure Function F(q) from Structure Factor S(q).
         This function generates the Q-axis based on the length of Sq and computes F(q) = Q * (S(q) - 1).
    [KR] 구조 인자 S(q)로부터 환원 구조 함수 F(q)를 계산합니다.
         Sq의 길이에 맞춰 Q축을 생성하고 F(q) = Q * (S(q) - 1) 공식을 적용합니다.

    Args:
        qmin (float):
            [EN] Minimum Q value (inverse Angstrom).
            [KR] Q 값의 최소값 (단위: 1/A).

        qmax (float):
            [EN] Maximum Q value (inverse Angstrom).
            [KR] Q 값의 최대값 (단위: 1/A).

        Sq (numpy.ndarray or list):
            [EN] 1D array of Structure Factor S(q) data.
            [KR] 구조 인자 S(q) 데이터가 담긴 1차원 배열입니다.

        qstep (float, optional):
            [EN] Step size for Q. Default is 0.01.
                 (Note: In this specific implementation using np.linspace, qstep is not directly used to determine the grid size, but is included for parameter consistency).
            [KR] Q 값의 간격. 기본값은 0.01입니다.
                 (참고: np.linspace를 사용하는 이 구현에서는 그리드 크기 결정에 직접 사용되지 않지만, 파라미터 일관성을 위해 포함되었습니다).

    Returns:
        tuple (numpy.ndarray, numpy.ndarray):
            [EN] A tuple containing (Fq, q_range).
                 - Fq: Calculated Reduced Structure Function F(q).
                 - q_range: The generated Q-axis values.
            [KR] (Fq, q_range) 튜플을 반환합니다.
                 - Fq: 계산된 환원 구조 함수 F(q).
                 - q_range: 생성된 Q축 값들.
    """
    # [EN] Generate Q axis evenly spaced between qmin and qmax matching the length of Sq
    # [KR] Sq의 데이터 길이에 맞춰 qmin과 qmax 사이의 등간격 Q축을 생성합니다.
    q_range = np.linspace(qmin, qmax, len(Sq), endpoint=False)

    # [EN] Calculate F(q) = Q * (S(q) - 1) and return with Q axis
    # [KR] F(q) = Q * (S(q) - 1) 계산 후 Q축과 함께 반환
    return (Sq - 1) * q_range, q_range


def cal_expGr_fft(q, Sq_or_Fq, rmin=0, rmax=100, rstep=0.01, is_Fq=False,
                  pad_mode="zero", low_q_mode="linear", extrapolate_type="linear",
                  return_padding=False):
    """
    [EN] Calculate G(r) from S(q) or F(q) using IFFT with low-q extrapolation
         and high-q padding.

         Low-Q extrapolation modes (low_q_mode):
           - "anchor" (default): F(q=0)=0 enforced (physically correct),
                                 straight line from (0,0) to (qmin, F(qmin)).
           - "linear": scipy interp1d extrapolation (legacy method),
                      uses the slope of the first two data points; may yield F(0)≠0.

         Nyquist check: if rmax > pi/qstep, F(q) is interpolated onto a finer q grid
                        and only the alias-free first half of the IFFT output is used.

         Visualization mode (return_padding=True):
           Additionally returns the three segments of the padded F(q) array
           used internally by the IFFT — useful for reproducing Figure 3 of the
           EZPDF paper (low-Q extrapolation in red, measured F(q) in black,
           high-Q zero padding in gray).

    [KR] IFFT를 사용하여 S(q) 또는 F(q)로부터 G(r)을 계산합니다.

         Low-Q 외삽 방식 (low_q_mode):
           - "anchor" (기본): F(q=0)=0을 강제 (물리적으로 정확),
                              (0,0)에서 (qmin, F(qmin))까지 직선.
           - "linear": scipy interp1d 외삽 (구버전 방식),
                      처음 두 점의 기울기 사용; F(0) ≠ 0 가능.

         Nyquist 체크: rmax > pi/qstep이면 F(q)를 더 촘촘한 q 그리드로 보간하고,
                       IFFT 출력의 alias-free 앞 절반만 사용합니다.

         시각화 모드 (return_padding=True):
           IFFT 내부에서 사용된 padded F(q)의 3개 구간을 추가로 반환합니다.
           EZPDF 논문 Figure 3 재현용 (Low-Q 외삽=빨강, 실험 F(q)=검정,
           High-Q zero padding=회색).

    Args:
        q (numpy.ndarray)       : Q-axis (1/Å)
        Sq_or_Fq (numpy.ndarray): S(q) or F(q) data
        rmin, rmax (float)      : real-space r range (Å)
        rstep (float)           : r step size (Å)
        is_Fq (bool)            : True → input is F(q), False → input is S(q)
        pad_mode (str)          : high-Q padding method
                                  'zero'     → fill with 0 (default)
                                  'decay'    → linear decay from last value to 0
                                  'constant' → extend last value
        low_q_mode (str)        : low-Q extrapolation method
                                  'anchor' → F(0)=0 enforced (default, physically correct)
                                  'linear' → scipy interp1d extrapolation (legacy)
        extrapolate_type (str)  : interpolation kind when low_q_mode='linear'
                                  ('linear', 'cubic', 'quadratic', etc.)
        return_padding (bool)   : if True, also return the three F(q) segments
                                  used for IFFT (for Figure 3 visualization)

    Returns:
        if return_padding=False (default):
            (r_list, gr) or (None, None) on error.

        if return_padding=True:
            (r_list, gr, padding_info)  or  (None, None, None) on error
            padding_info is a dict with keys:
                'q_low'         : q values of low-Q extrapolated region (q = 0 → qmin)
                'F_low'         : F(q) values of low-Q extrapolated region (RED in Figure 3)
                'q_exp'         : q values of measured experimental region (qmin → qmax)
                'F_exp'         : F(q) values of measured experimental region (BLACK)
                'q_pad'         : q values of high-Q padding region (qmax →)
                'F_pad'         : F(q) values of high-Q padding region (GRAY)
                'q_full_padded' : full Q axis fed to IFFT (0 → N × qstep)
                'F_full_padded' : full padded F(q) array fed to IFFT (concatenation of all 3 segments)
                'qstep'         : q-step actually used for IFFT (may differ from input
                                  if Nyquist interpolation was applied)
                'total_N'       : total number of points fed to IFFT
    """
    try:
        if len(q) < 2:
            print(f"Error in cal_expGr_fft: q array is too short (len={len(q)}) to calculate qstep.")
            return (None, None, None) if return_padding else (None, None)

        qstep_calc = q[1] - q[0]

        if qstep_calc == 0.0:
            print(f"Error in cal_expGr_fft: qstep is zero (q[1]==q[0]). Cannot divide by zero.")
            return (None, None, None) if return_padding else (None, None)

        num_point = int(np.ceil(q[0] / qstep_calc))

        if is_Fq:
            Fq = Sq_or_Fq
        else:
            Fq = (Sq_or_Fq - 1) * q

        pad_Fq = np.zeros(num_point)

        if q[0] > 0.0:
            if low_q_mode == "anchor":
                # [EN] Low-q extrapolation with F(q=0) = 0 enforced (anchor method).
                #      F(q) = q*(S(q)-1), so F(0) = 0 by definition regardless of S(0).
                #      A straight line from (0, 0) to (q[0], Fq[0]) fills the gap,
                #      which is mathematically consistent with the definition of F(q).
                # [KR] anchor 방식: F(0)=0을 강제하고 (qmin, F(qmin))까지 직선 연결.
                q_low = np.arange(num_point) * qstep_calc
                pad_Fq[:num_point] = (Fq[0] / q[0]) * q_low
            else:
                # [EN] Linear extrapolation (legacy method via scipy interp1d).
                #      Uses the slope of the first two data points to extrapolate down to q=0.
                #      May result in F(0) ≠ 0 (not physically guaranteed).
                # [KR] Linear 외삽 (구버전, scipy interp1d 사용).
                #      처음 두 점의 기울기로 q=0까지 외삽. F(0) ≠ 0 가능.
                f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate",
                                               kind=extrapolate_type)
                pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep_calc)


        # [EN] Save the three segments for visualization (BEFORE Nyquist interpolation)
        # [KR] Nyquist 보간 전 3구간을 시각화용으로 저장
        q_low_vis = np.arange(num_point) * qstep_calc
        F_low_vis = pad_Fq.copy()
        q_exp_vis = q.copy()
        F_exp_vis = Fq.copy()

        Fq_vals = np.append(pad_Fq, Fq)
        r_list = np.arange(rmin, rmax + rstep, rstep)

        # ------------------------------------------------------------------
        # Nyquist check:
        #   The IFFT produces total_point values; only the first half
        #   (indices 0 .. total_point//2) corresponds to physically valid
        #   positive-r values, giving r_nyquist = pi / qstep.
        #   If the requested rmax exceeds this limit, the second (aliased)
        #   half of the IFFT output would be used, introducing spurious
        #   oscillations at high r.
        #   Fix: interpolate F(q) onto a finer q grid so that
        #   pi / qstep_fine >= rmax before running the IFFT.
        # ------------------------------------------------------------------
        r_nyquist = np.pi / qstep_calc
        if rmax > r_nyquist:
            qstep_needed = np.pi / rmax
            q_orig = np.arange(len(Fq_vals)) * qstep_calc
            q_fine = np.arange(q_orig[0], q_orig[-1] + qstep_needed, qstep_needed)
            Fq_vals = np.interp(q_fine, q_orig, Fq_vals)
            qstep_calc = qstep_needed

        # --- High-Q padding (pad_mode: 'zero' / 'decay' / 'constant') ---
        num_point = len(Fq_vals)
        total_point = int(2 * np.pi / (rstep * qstep_calc))
        pad_len = total_point - num_point

        if pad_len > 0:
            last_val = Fq_vals[-1]
            if pad_mode == "decay":
                high_q_padding = np.linspace(last_val, 0, pad_len)
            elif pad_mode == "constant":
                high_q_padding = np.full(pad_len, last_val)
            else:  # "zero" (default)
                high_q_padding = np.zeros(pad_len)
            Fq_pad = np.concatenate((Fq_vals, high_q_padding))
        else:
            Fq_pad = Fq_vals
            high_q_padding = np.array([])

        # [EN] Save high-Q padding segment for visualization
        # [KR] High-Q 패딩 구간을 시각화용으로 저장
        q_pad_vis = q[-1] + qstep_calc + np.arange(len(high_q_padding)) * qstep_calc
        F_pad_vis = high_q_padding.copy() if len(high_q_padding) > 0 else np.array([])

        norm = total_point * qstep_calc * 2 / np.pi
        gr_full = norm * np.imag(np.fft.ifft(Fq_pad))

        # Use only the alias-free first half of the IFFT output
        half = total_point // 2
        rfine = np.arange(half) * rstep
        gr_fine = gr_full[:half]
        gr = np.interp(r_list, rfine, gr_fine)

        if return_padding:
            # [EN] Full padded F(q) array fed to IFFT (concatenation of all 3 segments)
            # [KR] IFFT에 실제로 들어간 전체 padded F(q) 배열 (3개 구간 합친 것)
            q_full_padded = np.arange(total_point) * qstep_calc
            F_full_padded = Fq_pad.copy()
            # print('q_full_padded = ', q_full_padded)
            # print('F_full_padded = ', F_full_padded)
            #
            np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/q_full_padded.txt', q_full_padded)
            np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/F_full_padded.txt', F_full_padded)

            padding_info = {
                'q_low':         q_low_vis,
                'F_low':         F_low_vis,
                'q_exp':         q_exp_vis,
                'F_exp':         F_exp_vis,
                'q_pad':         q_pad_vis,
                'F_pad':         F_pad_vis,
                'q_full_padded': q_full_padded,   # [EN/KR] Full Q axis (0 → N×qstep)
                'F_full_padded': F_full_padded,   # [EN/KR] Full padded F(q) fed to IFFT
                'qstep':         qstep_calc,
                'total_N':       total_point,
            }
            return r_list, gr, padding_info
        return r_list, gr

    except Exception as e:
        print(f"Error in cal_expGr_fft: {e}")
        return (None, None, None) if return_padding else (None, None)


def cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=100, rstep=0.01, pad_mode="zero",
                           low_q_mode="anchor", extrapolate_type="linear"):
    """
    [EN] Calculate G(r) directly from F(q) using IFFT.
         Same approach as cal_expGr_fft (anchor/linear low-q extrapolation, Nyquist check).
    [KR] F(q)로부터 직접 IFFT를 사용하여 G(r)을 계산합니다.
         cal_expGr_fft와 동일한 방식 (anchor/linear low-q 외삽, Nyquist 체크).

    Args:
        pad_mode (str)        : 'zero' (default) / 'decay' / 'constant'
        low_q_mode (str)      : 'anchor' (default, F(0)=0 enforced) / 'linear' (legacy)
        extrapolate_type (str): interpolation kind when low_q_mode='linear'
    """
    qstep = q[1] - q[0]
    num_point = int(np.ceil(q[0] / qstep))
    pad_Fq = np.zeros(num_point)

    if q[0] > 0.0:
        if low_q_mode == "anchor":
            # Low-q extrapolation with F(q=0) = 0 enforced (anchor method).
            q_low = np.arange(num_point) * qstep
            pad_Fq[:num_point] = (Fq[0] / q[0]) * q_low
        else:
            # Linear extrapolation (legacy method via scipy interp1d).
            f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate",
                                           kind=extrapolate_type)
            pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep)

    Fq_vals = np.append(pad_Fq, Fq)
    r_list = np.arange(rmin, rmax + rstep, rstep)

    # Nyquist check: interpolate onto finer q grid if rmax > pi/qstep
    r_nyquist = np.pi / qstep
    if rmax > r_nyquist:
        qstep_needed = np.pi / rmax
        q_orig = np.arange(len(Fq_vals)) * qstep
        q_fine = np.arange(q_orig[0], q_orig[-1] + qstep_needed, qstep_needed)
        Fq_vals = np.interp(q_fine, q_orig, Fq_vals)
        qstep = qstep_needed

    # --- High-Q padding (pad_mode: 'zero' / 'decay' / 'constant') ---
    num_point = len(Fq_vals)
    total_point = int(2 * np.pi / (rstep * qstep))
    pad_len = total_point - num_point

    if pad_len > 0:
        last_val = Fq_vals[-1]
        if pad_mode == "decay":
            high_q_padding = np.linspace(last_val, 0, pad_len)
        elif pad_mode == "constant":
            high_q_padding = np.full(pad_len, last_val)
        else:  # "zero" (default)
            high_q_padding = np.zeros(pad_len)
        Fq_pad = np.concatenate((Fq_vals, high_q_padding))
    else:
        Fq_pad = Fq_vals

    norm = total_point * qstep * 2 / np.pi
    gr_full = norm * np.imag(np.fft.ifft(Fq_pad))

    # Use only alias-free first half
    half = total_point // 2
    rfine = np.arange(half) * rstep
    gr_fine = gr_full[:half]
    gr = np.interp(r_list, rfine, gr_fine)

    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/pad_len.txt', pad_len)
    np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Fq_pad.txt', Fq_pad)

    return r_list, gr


def detect_header_lines(path, min_cols=2):
    """
    [EN] Automatically detect the number of header lines to skip.
    [KR] 데이터 파일에서 건너뛸 헤더(Header) 라인 수를 자동으로 감지합니다.
    """
    skip = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:  # [EN] Skip empty lines / [KR] 빈 줄 무시
                skip += 1
                continue
            parts = stripped.split()
            if stripped.startswith("#"):  # [EN] Skip comments / [KR] 주석(#) 무시
                skip += 1
                continue
            if len(parts) < min_cols:  # [EN] Skip if not enough columns / [KR] 열 개수 부족 시 무시
                skip += 1
                continue
            try:
                for p in parts[:min_cols]: float(p)  # [EN] Check if float / [KR] 실수형 변환 가능 여부 확인
                break
            except ValueError:
                skip += 1
                continue
    return skip


def load_qiq_file(path, min_cols=2, usecols=(0, 1)):
    """
    [EN] Load Q vs I(q) data from file using automatic header detection.
    [KR] 헤더 자동 감지 기능을 사용하여 파일에서 Q vs I(q) 데이터를 로드합니다.
    """
    skip = detect_header_lines(path, min_cols=min_cols)
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin1")  # [EN] Decode using latin1 to avoid errors / [KR] 인코딩 오류 방지
    buf = io.StringIO(text)
    data = np.loadtxt(buf, skiprows=skip, usecols=usecols)
    return data


# =============================================================================
# [Added Section] Whittaker-Henderson Smoothing Functions (12/2025 Added)
# =============================================================================

def smooth_whittaker(y, lambda_=1600.0, order=2):
    """
    [EN] Whittaker-Henderson smoothing algorithm.
         It minimizes: sum((y - z)^2) + lambda * sum((delta^order z)^2)
    [KR] Whittaker-Henderson 스무딩 알고리즘입니다.
         데이터 충실도(원본과의 차이)와 부드러움(미분값의 크기) 사이의 균형을 맞춥니다.

    Args:
        y (array): [EN] Input data to be smoothed / [KR] 스무딩할 입력 데이터
        lambda_ (float): [EN] Smoothing parameter (Larger = Smoother) / [KR] 스무딩 강도 (클수록 부드러워짐)
        order (int): [EN] Difference order (Usually 2) / [KR] 미분 차수 (보통 2 사용)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if order >= n:
        raise ValueError(f"Error: order={order} is too large for data length={n}. It must be less than {n}.")
    if lambda_ <= 0:
        raise ValueError(f"Error: lambda_ must be positive. Got {lambda_}.")

    try:
        DTD = make_dT_d(n, order)  # [EN] Create penalty matrix / [KR] 페널티 행렬 생성
        A = sp.eye(n, format='csc') + lambda_ * DTD  # A = I + lambda * D'D
        c, low = cho_factor(
            A.toarray())  # [EN] Cholesky factorization / [KR] 촐레스키 분해

        # [EN] Solve linear system Ay = z
        # [KR] 선형 연립방정식 Ay = z 를 풀어서 스무딩된 값 z를 구함
        return cho_solve((c, low), y)

    except LinAlgError as e:
        raise RuntimeError(f"Numerical error during smoothing: {e}. Consider lowering order or increasing lambda.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during smoothing: {e}. Check your lambda and order values.")


def make_dT_d(n, d):
    """
    [EN] Constructs the penalty matrix (D.T * D) for Whittaker smoothing.
    [KR] 스무딩을 위한 차분 행렬의 곱(D.T * D)을 생성합니다.
    """

    def diff_matrix(k, n):
        D = sp.eye(n, format='csc')
        for _ in range(k):
            D = D[1:] - D[:-1]  # [EN] Difference operator / [KR] 차분 연산자
        return D

    D = diff_matrix(d, n)
    return D.T @ D


def bandwidth_to_lambda(bandwidth, order):
    """
    Convert a Savitzky-Golay-like bandwidth to Whittaker lambda.
    """
    return (2 * factorial(order)) ** 2 / (bandwidth ** (2 * order))


def smooth_like_savitzky_golay(y, bandwidth, order):
    """
    Whittaker smoothing configured to mimic Savitzky-Golay behavior.
    """
    lambda_ = bandwidth_to_lambda(bandwidth, order)
    return smooth_whittaker(y, lambda_, order)


def noise_gain_to_lambda(gain, order):
    """
    Convert noise gain parameter to Whittaker lambda.
    """
    return (2 * factorial(order)) ** 2 / gain


def smooth_with_noise_gain(y, gain=1e-4, order=2):
    """
    Whittaker smoothing using a noise gain parameter.
    """
    lambda_ = noise_gain_to_lambda(gain, order)
    return smooth_whittaker(y, lambda_, order)