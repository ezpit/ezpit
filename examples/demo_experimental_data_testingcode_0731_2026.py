# ----------------------------------------------------------------------------------
# [EN] Path setup: add the EZPDF_code_version folder (parent of 'examples') to
#      sys.path so that 'losa' and 'proc' packages can be imported regardless of
#      the current working directory.
# [KR] 경로 설정: 'examples'의 상위 폴더(EZPDF_code_version)를 sys.path에 추가하여
#      실행 위치와 무관하게 'losa', 'proc' 패키지를 import할 수 있게 합니다.
# ----------------------------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt  # [EN] Library for plotting graphs / [KR] 그래프를 그리기 위한 라이브러리

# [EN] Fundamental package for numerical computation (Arrays)
# [KR] 수치 계산 및 배열 처리를 위한 핵심 패키지
import numpy as np

import ezpit.io as losa
import ezpit.processing as proc  # 파일 이름 바꾸었음.

# ----------------------------------------------------------------------------------
# Main Execution Block (메인 실행 블록: 실제 프로그램이 시작되는 곳)
# ----------------------------------------------------------------------------------

# [EN] File paths for atomic databases (Input files)
# [KR] 분석에 필요한 원자 데이터베이스 파일들의 경로를 설정합니다.
input_base = "C:/Users/gkwon/Pycharmprojects/ezpit/EZPDF_code_version/data/"
aff_element_file = input_base + "aff_elementonly.txt"  # [EN] Element symbols / [KR] 원소 기호 파일
aff_parm_file = input_base + "aff_parmonly.txt"  # [EN] Scattering parameters / [KR] 산란 인자 파라미터 파일

# [EN] File paths for Compton scattering databases
# [KR] 콤프턴 산란 보정을 위한 DB 파일 경로
compton_aff_element_file = input_base + "compton_element_only.txt"
compton_aff_parm_file = input_base + "compton_parameter_only.txt"
compton_atomnumber_file = input_base + "compton_atomicnumber.txt"

# [EN] Input data paths (Experiment and Background)
# [KR] 실제 실험 데이터(.chi 파일)와 배경 데이터 파일 경로
expqiq_input_base = "D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/"
bkgqiq_input_base = "D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/"

expqiq_data = expqiq_input_base + "A_CoPiITOglass_02142024_1-test_20240219-154304_696917_primary-dk_sub_image-0.chi"
bkgqiq_data = bkgqiq_input_base + "A_emptyquartzcap_0p5_20240219-122602_4e50c5_primary-dk_sub_image-0.chi"

# [EN] Parameters for Analysis (User settings)
# [KR] 분석 파라미터 설정 (사용자가 값을 바꾸며 테스트하는 곳)
# ----------------------------------------------------------------------------------
# [EN] Composition input — EZPDF_GUI_3 compatible. Several formats are accepted:
#        (1) dict           : {'Co': 38, 'O': 119, 'P': 20}
#        (2) compact string : "Co38O119P20"
#        (3) spaced string  : "Co 38 O 119 P 20"
#        (4) count-1 omitted: "SiO2"   (== {'Si': 1, 'O': 2})
#        (5) fractions      : "Li0.2Co0.36Mn0.37Ni0.07"  (== "Li20Co36Mn37Ni7")
#      For (1)-(4) (whole numbers) use convert_atom_names + group_atoms below.
#      For (5) (fractions) use composition_weights instead — see the commented
#      "Fractional composition example" block right after group_atoms.
# [KR] 조성 입력 — EZPDF_GUI_3 호환. 여러 형식을 지원합니다:
#        (1) 딕셔너리     : {'Co': 38, 'O': 119, 'P': 20}
#        (2) 붙여쓴 문자열 : "Co38O119P20"
#        (3) 공백 문자열   : "Co 38 O 119 P 20"
#        (4) 개수 1 생략   : "SiO2"   (== {'Si': 1, 'O': 2})
#        (5) 소수 조성     : "Li0.2Co0.36Mn0.37Ni0.07"  (== "Li20Co36Mn37Ni7")
#      (1)~(4) 정수 조성은 아래 convert_atom_names + group_atoms 사용.
#      (5) 소수 조성은 composition_weights 사용 — group_atoms 바로 뒤의
#      "Fractional composition example" 주석 블록 참고.
# ----------------------------------------------------------------------------------
composition = {
    "Co": 38,
    "O": 119,
    "P": 20,
}  # [EN] Chemical composition / [KR] 화학 조성
# composition = "Co38O119P20"                # [EN] String form (same result) / [KR] 문자열 (동일 결과)
# composition = "Co 38 O 119 P 20"           # [EN] Spaced string / [KR] 공백 구분 문자열
# composition = "SiO2"                        # [EN] Count of 1 omitted / [KR] 개수 1 생략
# composition = {'Co': 0.038, 'O': 0.119, 'P': 0.020}
# [EN] Fractional dict (auto-handled) / [KR] 소수 딜셔너리 (자동 처리)
# composition = "Li0.2Co0.36Mn0.37Ni0.07"    # [EN] Fractional string (auto-handled) / [KR] 소수 문자열 (자동 처리)
qmin = 0.6  # [EN] Minimum Q (float) / [KR] Q 최소값
qmax = 23  # [EN] Maximum Q (float) / [KR] Q 최대값
qstep = 0.01  # [EN] Q step size (float) / [KR] Q 간격
background_scale = 0.27  # [EN] Bkg scale factor (float) / [KR] 배경 제거 비율
qdamp = 0  # [EN] Resolution damping factor / [KR] 기기 해상도 감쇠 인자
poly_order = 7.208  # [EN] Polynomial order for correction (float) / [KR] F(q) 보정 다항식 차수
rpoly = np.pi * poly_order / qmax
# print("rpoly = ", rpoly)
rmin = 0  # [EN] Minimum r for G(r) / [KR] G(r) 계산 시작 거리
rmax = 20  # [EN] Maximum r for G(r) / [KR] G(r) 계산 끝 거리
rstep = 0.01  # [EN] r step size / [KR] 거리 간격
wavelength = 0.1665  # [EN] X-ray wavelength (Angstrom) / [KR] X선 파장
alpha = 3  # [EN] Compton recoil parameter / [KR] 콤프턴 반동 파라미터

# [EN] Compton Processing Steps
# [KR] 콤프턴 산란 계산 과정
# ----------------------------------------------------------------------------------
# [EN] Detect whether the composition is INTEGER or FRACTIONAL, and prepare both
#      the weight-based averaging (for S(q)) and an integer index list (for Compton,
#      which is inherently per-atom). For a fractional composition we scale every
#      element up to the smallest whole numbers only to build the Compton
#      atom_indices; the S(q) averaging still uses the exact fractional weights.
# [KR] 조성이 정수인지 소수인지 자동 감지합니다. S(q) 평균은 weight 기반으로,
#      Compton은 원자 단위이므로 정수 index 리스트를 준비합니다. 소수 조성이면
#      Compton용 atom_indices를 만들기 위해 원소들을 최소 정수배로만 스케일하고,
#      S(q) 평균 계산에는 정확한 소수 weight를 그대로 사용합니다.
# ----------------------------------------------------------------------------------
comp_parsed = losa.parse_composition(composition)  # dict (values may be float)
is_fractional = any(not float(v).is_integer() for v in comp_parsed.values())

# [EN] Per-unique-element names + weights (fraction-safe, for S(q) averaging)
# [KR] 고유 원소 이름 + weight (소수 지원, S(q) 평균용)
comp_names, comp_weights = losa.composition_weights(comp_parsed)

if is_fractional:
    # [EN] Build an integer index list for Compton by scaling to whole numbers.
    # [KR] Compton용 정수 index 리스트: 최소 정수배로 스케일하여 생성.
    from functools import reduce
    from math import gcd

    # scale fractions to integers (e.g. 0.038,0.119,0.020 -> 38,119,20 -> /gcd)
    max_dec = max(
        len(f"{v:.10f}".rstrip("0").split(".")[1]) if "." in f"{v:.10f}".rstrip("0") else 0 for v in comp_weights
    )
    scale = 10**max_dec
    int_counts = [int(round(v * scale)) for v in comp_weights]
    g = reduce(gcd, [c for c in int_counts if c > 0]) or 1
    int_counts = [c // g for c in int_counts]
    atom_names = [nm for nm, c in zip(comp_names, int_counts, strict=False) for _ in range(c)]
else:
    # [EN] Integer composition: expand directly into per-atom list.
    # [KR] 정수 조성: 원자별 리스트로 직접 확장.
    atom_names = losa.convert_atom_names(comp_parsed)

compton_atom_names = losa.load_atom_names(compton_aff_element_file)
compton_scat_parms = losa.load_scattering_factors(compton_aff_parm_file)
atom_unique_names, atom_counts, atom_indices = losa.group_atoms(atom_names)  # [EN] Group atoms / [KR] 원자 그룹화

# ----------------------------------------------------------------------------------
# [EN] Fractional composition example (optional)
#      NOTE: to actually run this, place it AFTER database_atom_names and
#      database_scat_factors are loaded (they are defined further below).
#      If your composition contains fractions (e.g. a doped/alloy sample written
#      the way it appears in a paper), convert_atom_names cannot expand it into
#      whole atoms. Use composition_weights instead, and pass 'weights=' to
#      cal_expSq. The scattering_factors must be ordered to match 'names'.
#      Fractions and their integer-scaled form give identical S(q):
#          "Li0.2Co0.36Mn0.37Ni0.07"  ==  "Li20Co36Mn37Ni7"
# [KR] 소수 조성 예제 (선택)
#      주의: 실제로 실행하려면 database_atom_names / database_scat_factors 가
#      로드된 뒤(아래쪽에 정의됨)에 배치하세요.
#      논문에 나오는 방식처럼 소수 조성(도핑/합금 시료 등)을 쓰는 경우,
#      convert_atom_names로는 정수 원자로 확장할 수 없습니다. 대신
#      composition_weights를 사용하고 cal_expSq에 'weights='를 전달하세요.
#      scattering_factors는 'names' 순서와 일치해야 합니다.
#      소수와 정수배 형태는 동일한 S(q)를 줍니다:
#          "Li0.2Co0.36Mn0.37Ni0.07"  ==  "Li20Co36Mn37Ni7"
#
#   frac_comp = "Li0.2Co0.36Mn0.37Ni0.07"
#   names, weights = losa.composition_weights(frac_comp)
#   scattering_factors_frac = losa.get_scattering_factors(
#       names, database_atom_names, database_scat_factors)
#   q, Iq, scaled_expIq, list_scaled_bkgIq, list_Sq, Sq, Fq, mean_sq_fi, sq_mean_fi, polynomial_for_sq = \
#       proc.cal_expSq(None, scattering_factors_frac, expqiq_data, bkgqiq_data,
#                      qmin=qmin, qmax=qmax, qstep=qstep,
#                      background_scale=background_scale, poly_order=poly_order,
#                      weights=weights)
# ----------------------------------------------------------------------------------

# [EN] Get Compton factors
# [KR] 콤프턴 인자 가져오기
compton_scat_form_factor, atomic_number = losa.get_compton_scattering_factors(
    atom_unique_names, compton_atom_names, compton_scat_parms
)

# [EN] Calculate Compton scattering intensity
# [KR] 콤프턴 산란 강도 계산
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
)

# [EN] Form Factor & S(q) Processing Steps
# [KR] 원자 형상 인자 준비 및 S(q) 계산 준비
database_atom_names = losa.load_atom_names(aff_element_file)
database_scat_factors = losa.load_scattering_factors(aff_parm_file)

atom_unique_names, atom_counts, atom_indices = losa.group_atoms(atom_names)

scattering_factors = losa.get_scattering_factors(atom_unique_names, database_atom_names, database_scat_factors)

# [EN] Calculate S(q) using the main function
# [KR] 메인 함수(cal_expSq)를 실행하여 S(q), F(q)를 계산합니다.
# [EN] Variables: q (Q-axis), Iq (Net intensity), Sq (Structure factor), Fq (Reduced structure function)
# [KR] 변수: q (Q축), Iq (최종 강도), Sq (구조 인자), Fq (환산 구조 함수)
# ----------------------------------------------------------------------------------
# [EN] For a FRACTIONAL composition, order scattering_factors to match comp_names
#      and pass weights=comp_weights. For an INTEGER composition, the usual
#      atom_indices path is used. Both give identical S(q).
# [KR] 소수 조성이면 scattering_factors를 comp_names 순서로 맞추고
#      weights=comp_weights를 전달합니다. 정수 조성이면 기존 atom_indices 경로.
#      두 방식 모두 동일한 S(q)를 줍니다.
# ----------------------------------------------------------------------------------
if is_fractional:
    scattering_factors_w = losa.get_scattering_factors(comp_names, database_atom_names, database_scat_factors)
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
        None,
        scattering_factors_w,
        expqiq_data,
        bkgqiq_data,
        qmin=qmin,
        qmax=qmax,
        qstep=qstep,
        background_scale=background_scale,
        poly_order=poly_order,
        return_Iq=False,
        weights=comp_weights,
    )
else:
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
    )


# np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Iq_baseline/sq_mean_fi.txt', sq_mean_fi)
# np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Iq_baseline/mean_sq_fi.txt', mean_sq_fi)
# np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Iq_baseline/normalized_intensity.txt', normalized_intensity)
# np.savetxt(
#     'D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Iq_baseline/normal_scattering_factor.txt',
#     normal_scattering_factor,
# )
# # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Iq_baseline/normalization_scale.txt', normalization_scale)
print("normalization_scale = ", normalization_scale)
# np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Iq_baseline/q.txt', q)
# np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/Iq_baseline/Iq.txt', Iq)

# ----------------------------------------------------------------------------------
# [Added] Apply Lorch Function & Direct G(r) Calculation
# [KR] Lorch 함수 적용 및 G(r) 직접 계산
# ----------------------------------------------------------------------------------
Fq_lorch = proc.apply_lorch_function(q, Fq)  # [EN] Apply Lorch to F(q) / [KR] F(q)에 Lorch 적용

# [EN] Calculate G(r) directly from F(q) (skipping S(q))
#      low_q_mode options for F(0) to F(qmin):
#        "anchor" → F(0)=0 enforced, straight line to (qmin, F(qmin))  [default]
#        "linear" → scipy interp1d extrapolation (legacy method)
# [KR] S(q)를 거치지 않고 F(q)에서 바로 G(r) 계산
#      F(0) ~ F(qmin) 외삽 옵션 (low_q_mode):
#        "anchor" → F(0)=0 강제, (qmin, F(qmin))까지 직선  [기본값]
#        "linear" → scipy interp1d 외삽 (구버전)
r_lorch, Gr_lorch = proc.cal_expGr_fft_from_Fq(q, Fq_lorch, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear")

# ----------------------------------------------------------------------------------
# [Added] Whittaker-Henderson Smoothing Application
# [KR] Whittaker-Henderson 스무딩 적용 (노이즈 제거)
# ----------------------------------------------------------------------------------
# [EN] Lambda: Smoothing strength (Typical: 100 ~ 10000)
# [KR] Lambda: 스무딩 강도 (보통 100 ~ 10000 사용)
whittaker_lambda = 1000.0
order = 2  # [EN] Order of difference / [KR] 미분 차수

# 1. [EN] Apply smoothing to F(q) / [KR] F(q)에 스무딩 적용
Fq_smoothed = proc.smooth_whittaker(Fq, lambda_=whittaker_lambda, order=order)

# 2. [EN] Calculate G(r) from smoothed F(q)
# [KR] 스무딩된 F(q)를 사용하여 G(r) 계산
r_smooth, Gr_from_smoothFq = proc.cal_expGr_fft_from_Fq(
    q, Fq_smoothed, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear"
)

# [EN] Calculate G(r) from RAW F(q) for comparison
# [KR] 비교를 위해 "원본(Raw) F(q)"로 만든 G(r)도 계산
r_raw, Gr_from_rawFq = proc.cal_expGr_fft_from_Fq(q, Fq, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear")

# ----------------------------------------------------------------------------------
# Plotting Results (결과 그래프 출력)
# ----------------------------------------------------------------------------------
num_atom = len(atom_indices)

# print('atom_indices = ', atom_indices)

# [EN] Save intermediate results to text files
# [KR] 중간 계산 결과들을 각각 텍스트 파일로 저장합니다.
np.savetxt(expqiq_input_base + "q_scaled_expIq.iq", np.column_stack([q, scaled_expIq]))
np.savetxt(
    expqiq_input_base + "q_list_scaled_bkgIq.iq",
    np.column_stack([q, list_scaled_bkgIq]),
)
np.savetxt(expqiq_input_base + "bkg_subtraced_expqIq.iq", np.column_stack([q, Iq]))
np.savetxt(expqiq_input_base + "notnormalized_qSq.sq", np.column_stack([q, list_Sq]))
np.savetxt(
    expqiq_input_base + "polynomial_qpolySq.sq",
    np.column_stack([q, polynomial_for_sq]),
)
np.savetxt(expqiq_input_base + "normalized_qSq.sq", np.column_stack([q, Sq]))
np.savetxt(expqiq_input_base + "qFq.fq", np.column_stack([q, Fq]))

# Figure 1: Background Subtraction Check / 배경 제거 확인
plt.figure(1)
plt.plot(q, scaled_expIq, label="Exp (Raw)")  # [EN] Raw Exp Data / [KR] 원본 실험 데이터
plt.plot(q, list_scaled_bkgIq, label="Bkg * Scale")  # [EN] Scaled Background / [KR] 스케일링된 배경
plt.plot(q, Iq, label="Net I(q)")  # [EN] Net Intensity / [KR] 배경 제거된 데이터
plt.xlabel("q (1/A)")
plt.ylabel("I(q) [Raw Counts]")
plt.grid()
plt.legend()

# Figure 2: Net Intensity I(q) / 최종 I(q)
plt.figure(2)
plt.plot(q, Iq, label="Net I(q)")
plt.xlabel("q (1/A)")
plt.ylabel("I(q)")
plt.grid()
plt.legend()

# Figure 3: S(q) Comparison / S(q) 비교
plt.figure(3)
plt.plot(q, list_Sq, label="Standard S(q)")  # [EN] Standard S(q) / [KR] 기본 S(q)
plt.plot(q, Sq, label="Poly Corrected S(q)")  # [EN] Polynomial Corrected / [KR] 다항식 보정된 S(q)
plt.xlabel("q (1/A)")
plt.ylabel("S(q)")
plt.grid()
plt.legend()

plt.figure(31)
plt.plot(q, Sq, label="Poly Corrected S(q)")
plt.xlabel("q (1/A)")
plt.ylabel("S(q)")
plt.grid()
plt.legend()

plt.figure(32)
plt.plot(q, list_Sq, label="Standard S(q)")  # [EN] Standard S(q) / [KR] 기본 S(q)
plt.plot(q, polynomial_for_sq, label="Polynomial for S(q)")  # [EN] Polynomial Corrected / [KR] 다항식 보정된 S(q)
plt.xlabel("q (1/A)")
plt.ylabel("S(q)")
plt.grid()
plt.legend()

# Figure 4: F(q) Comparison (Raw vs Smoothed vs Lorch) / F(q) 비교 (중요!)
plt.figure(4)
plt.plot(q, Fq, label="F(q)")  # [EN] Original F(q) / [KR] 원본 F(q)
# plt.plot(q, Fq_smoothed, label='F(q)_WH smoothed')  # [EN] Whittaker Smoothed / [KR] Whittaker 스무딩된 F(q)
# plt.plot(q, Fq_lorch, label='F(q)_lorch')  # [EN] Lorch Applied / [KR] Lorch 적용된 F(q)
plt.xlabel("q (1/A)")
plt.ylabel("F(q)")
plt.grid()
plt.legend()

# G(r) Calculation Comparison (Different Padding Modes)
# [EN] Comparing 3 padding modes (decay, constant, zero) with anchor low-Q extrapolation
# [KR] 다양한 패딩 모드(Decay, Constant, Zero)에 따른 G(r) 비교 (anchor low-Q 외삽 사용)
r3, Gr3 = proc.cal_expGr_fft(q, Sq, rmin, rmax, rstep, pad_mode="decay", low_q_mode="linear")
r4, Gr4 = proc.cal_expGr_fft(q, Sq, rmin, rmax, rstep, pad_mode="constant", low_q_mode="linear")
r5, Gr5 = proc.cal_expGr_fft(q, Sq, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear")

np.savetxt(expqiq_input_base + "r3gr3_decay.txt", np.column_stack([r3, Gr3]))
np.savetxt(expqiq_input_base + "r4gr4_constant.txt", np.column_stack([r4, Gr4]))
np.savetxt(expqiq_input_base + "r5gr5_zero.txt", np.column_stack([r5, Gr5]))

# ----------------------------------------------------------------------------------
# Figure 3 (EZPDF paper) - Data extension for IFFT-based G(r) calculation
# [EN] Visualize the three segments fed to IFFT:
#      Low-Q extrapolation (RED) + measured F(q) (BLACK) + high-Q zero padding (GRAY)
# [KR] IFFT에 들어가는 3개 구간 시각화:
#      Low-Q 외삽 (빨강) + 실험 F(q) (검정) + High-Q zero padding (회색)
# ----------------------------------------------------------------------------------
# [EN] Get padding info using return_padding=True option
# [KR] return_padding=True 옵션으로 padding 정보 함께 받기
_, _, padding_info = proc.cal_expGr_fft(
    q, Sq, rmin, rmax, rstep, pad_mode="zero", low_q_mode="linear", return_padding=True
)

# [EN] X-axis display range: only ~7 Å⁻¹ beyond qmax (zero pad is too long to show fully)
# [KR] x축 표시 범위: qmax 이후 약 7 Å⁻¹ 까지만 (zero padding 전체는 너무 길어서 잘라냄)
pad_show_qrange = 7.0
pad_mask = padding_info["q_pad"] <= (qmax + pad_show_qrange)

plt.figure(99, figsize=(12, 5.5))
# [EN] Low-Q extrapolation region (RED, Figure 3 emphasis)
# [KR] Low-Q 외삽 영역 (빨간색, Figure 3 강조)
plt.plot(
    padding_info["q_low"],
    padding_info["F_low"],
    "r-",
    linewidth=2.5,
    label="Low-Q extrapolation (q=0 → qmin)",
)
plt.plot(
    [qmin], [padding_info["F_exp"][0]], "ro", markersize=8, zorder=5
)  # [EN] Anchor connection point / [KR] anchor 연결점

# [EN] Measured experimental F(q) region (BLACK)
# [KR] 실험 F(q) 영역 (검은색)
plt.plot(
    padding_info["q_exp"],
    padding_info["F_exp"],
    "k-",
    linewidth=1.0,
    label="Measured F(q) (qmin → qmax)",
)

# [EN] High-Q zero padding region (GRAY, partial)
# [KR] High-Q zero padding 영역 (회색, 일부만)
plt.plot(
    padding_info["q_pad"][pad_mask],
    padding_info["F_pad"][pad_mask],
    color="gray",
    linewidth=2.0,
    label="High-Q zero padding (qmax → N pts)",
)

# [EN] Region boundary lines
# [KR] 영역 경계선
plt.axvline(x=0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
plt.axvline(x=qmin, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
plt.axvline(x=qmax, color="blue", linestyle="--", linewidth=1.2, alpha=0.7)
plt.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)

# [EN] Info text box (qstep, total N, points per region)
# [KR] 정보 텍스트 박스 (qstep, 전체 N, 영역별 포인트 수)
info_text = (
    f"qstep = {padding_info['qstep']:.4f}\n"
    f"N = 2π/(rstep·qstep) = {padding_info['total_N']:,}\n"
    f"Low-Q   : {len(padding_info['q_low']):,} pts\n"
    f"Exp F(q): {len(padding_info['q_exp']):,} pts\n"
    f"Zero pad: {len(padding_info['q_pad']):,} pts"
)
plt.text(
    0.985,
    0.05,
    info_text,
    transform=plt.gca().transAxes,
    fontsize=8.5,
    ha="right",
    va="bottom",
    bbox={
        "boxstyle": "round,pad=0.5",
        "facecolor": "lightyellow",
        "edgecolor": "gray",
        "alpha": 0.95,
    },
    family="monospace",
)

plt.title("Figure 3. Data extension for IFFT-based G(r) calculation")
plt.xlabel("q (1/Å)")
plt.ylabel("F(q)")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.25)
plt.xlim(-1, qmax + pad_show_qrange + 1)
plt.tight_layout()

# ----------------------------------------------------------------------------------
# Figure 3B - Full padded F(q) array (everything fed to IFFT)
# [EN] Show the COMPLETE padded F(q) including all zero-padding up to N×qstep.
#      This is the actual array passed to numpy.fft.ifft().
# [KR] IFFT에 들어가는 전체 padded F(q) 표시 (zero padding 전부 포함).
#      numpy.fft.ifft()에 실제로 전달되는 배열입니다.
# ----------------------------------------------------------------------------------
plt.figure(100, figsize=(12, 5.5))
# [EN] Full padded F(q): single continuous line over all N points
# [KR] 전체 padded F(q): N개 포인트 전체에 걸친 연속 선
plt.plot(
    padding_info["q_full_padded"],
    padding_info["F_full_padded"],
    "b-",
    linewidth=0.6,
    label=f"Full padded F(q)  (N = {padding_info['total_N']:,} pts)",
)

# [EN] Overlay region boundaries for clarity
# [KR] 영역 경계 표시
plt.axvline(
    x=qmin,
    color="red",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label=f"qmin = {qmin}",
)
plt.axvline(
    x=qmax,
    color="green",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label=f"qmax = {qmax}",
)
nyquist_q = np.pi / padding_info["qstep"]
plt.axvline(
    x=nyquist_q,
    color="purple",
    linestyle=":",
    linewidth=1.0,
    alpha=0.7,
    label=f"π/qstep = {nyquist_q:.1f} (Nyquist limit)",
)
plt.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)

# [EN] Region shading
# [KR] 영역 음영
plt.axvspan(0, qmin, alpha=0.12, color="red", label="_nolegend_")
plt.axvspan(qmin, qmax, alpha=0.06, color="black", label="_nolegend_")
plt.axvspan(
    qmax,
    padding_info["q_full_padded"][-1],
    alpha=0.08,
    color="gray",
    label="_nolegend_",
)

plt.title("Figure 3B. Full padded F(q) array fed to IFFT")
plt.xlabel("q (1/Å)")
plt.ylabel("F(q)")
plt.legend(loc="upper right", fontsize=9)
plt.grid(True, alpha=0.25)
plt.xlim(-5, padding_info["q_full_padded"][-1] + 5)
plt.tight_layout()

plt.figure(71)
plt.plot(r_lorch, Gr_lorch, label="Gr_lorch_zero")  # [EN] Result of Lorch / [KR] Lorch 적용 결과
plt.plot(r5, Gr5, label="zero")
plt.title("G(r) after Lorch")
plt.xlabel("r (A)")
plt.ylabel("G(r)")
plt.grid()
plt.legend()

plt.figure(72)
plt.plot(r_raw, Gr_from_rawFq, label="Gr_WHsmoothed_zero")  # [EN] Result of Smoothing / [KR] Whittaker 스무딩 결과
plt.plot(r5, Gr5, label="zero")
plt.title("G(r) after Lorch")
plt.xlabel("r (A)")
plt.ylabel("G(r)")
plt.grid()
plt.legend()

plt.figure(7)
plt.plot(r3, Gr3, label="decay")
plt.plot(r4, Gr4, label="constant")
plt.plot(r5, Gr5, label="zero")

plt.title("G(r) Comparison")
plt.xlabel("r (A)")
plt.ylabel("G(r)")
plt.grid()
plt.legend()
plt.show()  # [EN] Show all plots / [KR] 모든 그래프 화면에 출력
