# model/math_functions.py

import numpy as np
from scipy.spatial.distance import cdist
from scipy import interpolate
from ezpit.gui.model.elem_data import ElementData
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import scipy.sparse as sp
from math import factorial
import io

data = ElementData()


def create_atom_distance_matrix(atom_positions):
    distance_matrix = cdist(atom_positions, atom_positions)
    return distance_matrix


def __cal_fi(scat_values, q):
    """
    Calculate scattering form factor (Vectorized)
    q can be a scalar or a numpy array.
    """
    # [Vectorization] q가 배열일 경우 자동으로 배열 연산 수행
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
    Calculate Compton scattering form factor (Vectorized support)
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
                    atomic_number, qmin=0.2, qmax=20, qstep=0.1, wavelength=0.1665, alpha=3):

    #============================================================================
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

    # parameters for Breit-Dirac recoil factor
    me = 9.109534e-31
    C = 2.99792458e18  # 8+10 m to A
    h = 6.62607015e-14  # -34+20 , m^2 to A^2
    part_A = 2.0 * h * wavelength / me / C

    num_atom = len(atom_indices)
    num_fact = len(compton_scattering_factors)
    q_range = np.arange(qmin, qmax, qstep)

    # [Optimization Note] This loop is kept as is, but can be vectorized if needed.
    # Typically Compton calculation is done once and is fast enough.
    list_compton_scat = []
    compton_scat = 0.0

    for q in q_range:
        atomic_number_sum = 0.0
        ffit_sum = 0.0
        #fi2_sum = 0.0
        part_B = (q / (4.0 * np.pi)) ** 2.0
        BD_recoil_fact = (part_A * part_B + 1) ** (-alpha)  # Breit-Dirac recoil factor
        list_fi = []
        for k in range(num_fact):
            list_fi.append(__cal_compton_fi(compton_scat_parms[atomic_number[k] - 1], q))
        list_fi = np.asarray(list_fi)
        for i, idx in enumerate(atom_indices):
            fi = list_fi[idx]
            atomic_number_sum = atomic_number_sum + atomic_number[idx]
            ffit_sum = ffit_sum + fi  # I_inc = Z - F_fit (Balyuzi 1975); F_fit is already (Z - I_inc)
            #fi2_sum = fi2_sum + fi   #removed due to a wwrong equation (fi ** 2) / atomic_number[idx]
        compton_scat = BD_recoil_fact * (1 / num_atom * atomic_number_sum - 1 / num_atom * ffit_sum)
        list_compton_scat.append(compton_scat)
    return q_range, list_compton_scat


def cal_Iq(atom_indices, scattering_factors, atom_distance_matrix,
           qmin=0.5, qmax=20, qstep=0.05):
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


def cal_expSq(atom_indices, scattering_factors, expqiq, bkgqiq, qmin=0, qmax=25, qstep=0.01,
              background_scale=1.1, poly_order=11.0, return_Iq=False):
    """
    [VECTORIZED] Optimized version for Experimental S(q) calculation.
    Removed 'for q in q_range' loop for significant speedup.
    """
    # -------------------------------------------------------------------
    # 1. Experimental Data 처리
    # -------------------------------------------------------------------
    if isinstance(expqiq, str):
        data_exp = load_qiq_file(expqiq, min_cols=2, usecols=(0, 1))
        exp_q = data_exp[:, 0]
        exp_Iq = data_exp[:, 1]
    else:
        exp_q = expqiq[0]
        exp_Iq = expqiq[1]

    # -------------------------------------------------------------------
    # 2. Background Data 처리
    # -------------------------------------------------------------------
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
        bkg_Iq = np.zeros_like(exp_Iq)

    # q-grid 구성 및 데이터 보간
    q_range = np.arange(qmin, qmax, qstep)

    # exp_q 기준으로 보간
    scaled_expIq = np.interp(q_range, exp_q, exp_Iq)
    scaled_bkgIq = np.interp(q_range, exp_q, bkg_Iq)

    list_scaled_bkgIq = background_scale * scaled_bkgIq
    list_Iq = scaled_expIq - list_scaled_bkgIq

    # -------------------------------------------------------------------
    # 3. Form Factor Calculation (Vectorized)
    # -------------------------------------------------------------------
    num_atom = len(atom_indices)

    # (1) 모든 Unique Element(num_fact)에 대해 전체 q_range의 Form Factor 미리 계산
    # 결과 shape: (num_fact, len(q_range))
    # __cal_fi 함수가 numpy array 연산을 지원하므로 for loop 없이 한 번에 계산
    list_fi_all_q = np.array([__cal_fi(sf, q_range) for sf in scattering_factors])

    # (2) 각 Atom Index에 맞는 Form Factor 배열로 확장
    # atom_indices를 이용해 broadcast. 결과 shape: (num_atom, len(q_range))
    atom_fi_all_q = list_fi_all_q[atom_indices]

    # (3) <f> 및 <f^2> 계산
    # axis=0 (Atom 방향)으로 합계 계산하여 (len(q_range), ) 크기의 1차원 배열 생성
    sum_fi = np.sum(atom_fi_all_q, axis=0)  # Sum of f
    sum_fi_sq = np.sum(atom_fi_all_q ** 2, axis=0)  # Sum of f^2

    sq_mean_fi = (sum_fi / num_atom) ** 2  # <f>^2
    mean_sq_fi = sum_fi_sq / num_atom  # <f^2>

    # S(q) 계산 (배열 연산)
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

    normalized_intensity = list_Iq / sq_mean_fi          # I / <f>^2   (sq_mean_fi = <f>^2)
    normal_scattering_factor = mean_sq_fi / sq_mean_fi   # <f^2> / <f>^2 (mean_sq_fi = <f^2>)
    normalization_scale = (np.dot(normalized_intensity, normal_scattering_factor)
                           / np.dot(normalized_intensity, normalized_intensity))

    # print(f"[DEBUG2] sum_fi[0]={sum_fi[0]:.4f}, sum_fi_sq[0]={sum_fi_sq[0]:.4f}")
    # print(f"[DEBUG2] sq_mean_fi[0]={sq_mean_fi[0]:.4f}, mean_sq_fi[0]={mean_sq_fi[0]:.4f}")
    # print(f"[DEBUG2] scattering_factors.shape={np.array(scattering_factors).shape}")
    # print(f"[DEBUG2] atom_fi_all_q.shape={atom_fi_all_q.shape}")
    # print(f"[DEBUG2] list_fi_all_q.shape={list_fi_all_q.shape}")
    # print(f"[DEBUG2] normalized_intensity[0]={normalized_intensity[0]:.6f}, normal_scattering_factor[0]={normal_scattering_factor[0]:.6f}")

    list_Sq = normalization_scale * normalized_intensity - normal_scattering_factor + 1.0
    # print(f"[DEBUG] normalization_scale={normalization_scale}, list_Sq range={list_Sq.min():.3f} to {list_Sq.max():.3f}, num_atom={num_atom}")

    # F(q) 공간 다항 적합(Vandermonde, 상수항 제거)
    qmax_for_sq = float(np.max(q_range))

    def _poly_for_sq_at_order(order_int: int):
        order_int = max(int(order_int), 1)
        qscaled = q_range / qmax_for_sq
        V = np.vander(qscaled, order_int + 1)
        A = V[:, :-1]
        b = q_range * (list_Sq - 1.0)
        p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        vand2 = np.vander([qmax_for_sq], order_int + 1)[0, :-1]
        new_p = p / vand2
        final_p = np.poly1d(new_p)
        return final_p(q_range)

    # Poly order 처리
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

    # 정규화 & F(q)
    norm_list_Sq = list_Sq - polynomial_for_sq
    list_Fq = q_range * (norm_list_Sq - 1.0)

    if return_Iq:
        return (q_range, list_Iq, scaled_expIq, list_scaled_bkgIq,
                list_Sq, norm_list_Sq, list_Fq, mean_sq_fi, sq_mean_fi)
    else:
        return (q_range, list_Iq, scaled_expIq, list_scaled_bkgIq,
                list_Sq, norm_list_Sq, list_Fq, mean_sq_fi, sq_mean_fi)

    normalized_intensity = list_Iq / sq_mean_fi          # I / <f>^2   (sq_mean_fi = <f>^2)
    normal_scattering_factor = mean_sq_fi / sq_mean_fi   # <f^2> / <f>^2 (mean_sq_fi = <f^2>)
    normalization_scale = (np.dot(normalized_intensity, normal_scattering_factor)
                           / np.dot(normalized_intensity, normalized_intensity))

    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/sq_mean_fi.txt', sq_mean_fi)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/mean_sq_fi.txt', mean_sq_fi)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/normalized_intensity.txt', normalized_intensity)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/normal_scattering_factor.txt', normal_scattering_factor)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/normalization_scale.txt', normalization_scale)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/scaled_experimental_I(q).txt', normalization_scale * list_Iq )


def cal_expGr_fft(q, Sq_or_Fq, rmin=0, rmax=100, rstep=0.01,
                  extrapolate_type="linear", is_Fq=False, pad_mode="zero"):
    """
    Calculates G(r) from S(q) or F(q) using FFT with low-q extrapolation
    and high-q padding.

    Low-q extrapolation:
        The q < qmin region is filled by scipy interp1d extrapolation
        (extrapolate_type, default 'linear'). This uses the slope of the data
        near qmin and does NOT force F(q=0)=0, matching the EZPDF GUI behaviour.

    Nyquist check:
        The IFFT yields total_point values, but only the alias-free first half
        (r_nyquist = pi / qstep) is physically valid. If rmax exceeds r_nyquist,
        F(q) is interpolated onto a finer q grid before the IFFT, and only the
        first half of the output is used.

    High-q padding (pad_mode):
        'zero'     -> pad with zeros (default; used by the EZPDF GUI)
        'decay'    -> linear decay from the last F(q) value down to 0
        'constant' -> extend the last F(q) value
        The two non-zero modes are provided for completeness / experimentation
        and are not exposed in the GUI; callers must request them explicitly.

    Returns: (r_list, gr) or (None, None) on error.
    """
    try:
        if len(q) < 2:
            print(f"Error in cal_expGr_fft: q array is too short (len={len(q)}) to calculate qstep.")
            return None, None

        qstep_calc = q[1] - q[0]  # Renamed to avoid confusion with parameter if any

        if qstep_calc == 0.0:
            print(f"Error in cal_expGr_fft: qstep is zero (q[1]==q[0]). Cannot divide by zero.")
            return None, None

        num_point = int(np.ceil(q[0] / qstep_calc))

        if is_Fq:
            Fq = Sq_or_Fq
        else:
            Fq = (Sq_or_Fq - 1) * q

        # --- Low-q extrapolation (linear interp1d; F(0) != 0 allowed) ---
        pad_Fq = np.zeros(num_point)

        if q[0] > 0.0:
            f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate", kind=extrapolate_type)
            pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep_calc)

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

        num_point = len(Fq_vals)
        total_point = int(2 * np.pi / (rstep * qstep_calc))
        pad_len = total_point - num_point

        # --- High-q padding (pad_mode) ---
        if pad_len > 0:
            last_val = Fq_vals[-1]
            if pad_mode == "decay":
                # Linear decay from the last value down to 0 across the pad region.
                high_q_padding = np.linspace(last_val, 0.0, pad_len)
            elif pad_mode == "constant":
                # Extend the last value across the pad region.
                high_q_padding = np.full(pad_len, last_val)
            else:
                # Default "zero" padding (GUI behaviour).
                high_q_padding = np.zeros(pad_len)
            Fq_pad = np.concatenate((Fq_vals, high_q_padding))
        else:
            Fq_pad = Fq_vals

        norm = total_point * qstep_calc * 2 / np.pi
        gr_full = norm * np.imag(np.fft.ifft(Fq_pad))

        # Use only the alias-free first half of the IFFT output.
        half = total_point // 2
        rfine = np.arange(half) * rstep
        gr_fine = gr_full[:half]
        gr = np.interp(r_list, rfine, gr_fine)

        return r_list, gr

    except Exception as e:
        print(f"Error in cal_expGr_fft: {e}")
        return None, None


def cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=100, rstep=0.01,
                           extrapolate_type="linear"):
    qstep = q[1] - q[0]
    num_point = int(np.ceil(q[0] / qstep))
    pad_Fq = np.zeros(num_point)

    if q[0] > 0.0:
        f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate",
                                       kind=extrapolate_type)
        pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep)

    Fq_vals = np.append(pad_Fq, Fq)
    r_list = np.arange(rmin, rmax + rstep, rstep)

    num_point = len(Fq_vals)
    total_point = int(2 * np.pi / (rstep * qstep))
    Fq_pad = np.pad(Fq_vals, (0, total_point - num_point), mode="constant")

    norm = total_point * qstep * 2 / np.pi
    gr_fine = norm * np.imag(np.fft.ifft(Fq_pad))
    rfine = np.arange(total_point) * rstep
    gr = np.interp(r_list, rfine, gr_fine)

    return r_list, gr


def cal_fq(qmin, qmax, Sq, qstep=0.01):
    q_range = np.linspace(qmin, qmax, len(Sq), endpoint=False)
    return (Sq - 1) * q_range, q_range


def get_aff_scattering_factors(atom_names):
    return data.get_aff_scattering_factors(atom_names)


def get_compton_scattering_factors(atom_names):
    return data.get_compton_scattering_factors(atom_names)


def get_compton_parameter_only():
    return data.get_compton_parameter_only()


def detect_header_lines(path, min_cols=2):
    skip = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                skip += 1
                continue
            parts = stripped.split()
            if stripped.startswith("#"):
                skip += 1
                continue
            if len(parts) < min_cols:
                skip += 1
                continue
            try:
                for p in parts[:min_cols]:
                    float(p)
                break
            except ValueError:
                skip += 1
                continue
    return skip


def load_qiq_file(path, min_cols=2, usecols=(0, 1)):
    skip = detect_header_lines(path, min_cols=min_cols)
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin1")
    buf = io.StringIO(text)
    data = np.loadtxt(buf, skiprows=skip, usecols=usecols)
    return data


def smooth_whittaker(y, lambda_=1600.0, order=2):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if order >= n:
        raise ValueError(f"Error: order={order} is too large for data length={n}. It must be less than {n}.")
    if lambda_ <= 0:
        raise ValueError(f"Error: lambda_ must be positive. Got {lambda_}.")

    try:
        DTD = make_dT_d(n, order)
        A = np.eye(n) + lambda_ * DTD
        c, low = cho_factor(A)
        return cho_solve((c, low), y)
    except LinAlgError as e:
        raise RuntimeError(f"Numerical error during smoothing: {e}. Consider lowering order or increasing lambda.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during smoothing: {e}. Check your lambda and order values.")


def make_dT_d(n, d):
    def diff_matrix(k, n):
        D = sp.eye(n, format='csc')
        for _ in range(k):
            D = D[1:] - D[:-1]
        return D

    D = diff_matrix(d, n)
    return D.T @ D


def bandwidth_to_lambda(bandwidth, order):
    return (2 * factorial(order)) ** 2 / (bandwidth ** (2 * order))


def smooth_like_savitzky_golay(y, bandwidth, order):
    lambda_ = bandwidth_to_lambda(bandwidth, order)
    return smooth_whittaker(y, lambda_, order)


def noise_gain_to_lambda(gain, order):
    return (2 * factorial(order)) ** 2 / gain


def smooth_with_noise_gain(y, gain=1e-4, order=2):
    lambda_ = noise_gain_to_lambda(gain, order)
    return smooth_whittaker(y, lambda_, order)


def batch_smooth_whittaker(y_2d, lambda_=1600.0, order=2):
    return np.array([smooth_whittaker(row, lambda_, order) for row in y_2d])
