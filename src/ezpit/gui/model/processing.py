# model/processing.py

import numpy as np

from ezpit.elem_tables import (
    get_aff_scattering_factors,
    get_compton_parameter_only,
    get_compton_scattering_factors,
)
from ezpit.gui.model.helpers import (
    composition_weights,
    extract_data,
    group_atoms,
    parse_composition,
)
from ezpit.processing import (
    cal_expGr_fft,
    cal_expSq,
    cal_fq,
    cal_Sq,
    compton_cal_exp,
    create_atom_distance_matrix,
    smooth_whittaker,
)


def get_expSq(exp_data, control_panel, multiple_graphs=False):
    """Manipulates the data to get the appropriate S(q) and F(q) data from the experimental data."""
    bkg_x = None
    bkg_y = None
    background_scale = 0

    pdf_parameters = control_panel.get_pdf_parameters()
    basic_parameters = control_panel.get_basic_parameters()

    # [NEW] Extract q_step from control panel
    try:
        q_step = float(pdf_parameters.get("qstep", 0.01))
    except (ValueError, TypeError):
        q_step = 0.01

    if q_step <= 0:
        q_step = 0.01

    # -------------------------------------------------------------------------
    # 조성 처리: 원소별 조성량을 그대로 가중치로 사용
    # -------------------------------------------------------------------------
    # 1. 조성 문자열 파싱 -> Dictionary ({'Co':38, 'O':119, 'P':1})
    composition_dict = parse_composition(basic_parameters["composition"])
    # print(f"[DEBUG3] raw composition = {repr(basic_parameters['composition'])}")
    # print(f"[DEBUG3] composition_dict = {dict(composition_dict)}")

    # 2. 고유 원소 이름과 그 조성량을 같은 순서로 추출.
    #    원자를 개수만큼 나열하지 않으므로 조성 숫자가 커져도 비용이 늘지 않고,
    #    Li0.2Co0.36Mn0.37Ni0.07 같은 소수 조성도 그대로 처리된다.
    #    <f>, <f^2>는 총합으로 나눈 평균이므로 모든 원소에 같은 배수를 곱해도
    #    결과는 같다 (Li0.2Co0.36Mn0.37Ni0.07 == Li20Co36Mn37Ni7).
    unique_atom_names, atom_weights = composition_weights(composition_dict)

    # 3. 산란 인자는 고유 원소 이름 순서와 정확히 일치해야 한다.
    scattering_factors = get_aff_scattering_factors(unique_atom_names)

    # 4. atom_indices는 하위 호환을 위해 유지 (정수 조성일 때만 의미가 있으며,
    #    cal_expSq는 atom_weights가 주어지면 이 값을 사용하지 않는다).
    atom_indices = np.arange(len(unique_atom_names))
    # -------------------------------------------------------------------------

    unit_type = basic_parameters["data_format"]
    wavelength = float(basic_parameters["wavelength"])

    if unit_type == "2theta":
        theta_rad = np.radians(exp_data[0] / 2)
        exp_data[0] = (4 * np.pi * np.sin(theta_rad)) / wavelength
    elif unit_type == "q_nmn":
        exp_data[0] = exp_data[0] * 0.1

    qmin = pdf_parameters["qmin"]
    qmax = pdf_parameters["qmax"]
    background_scale = pdf_parameters["bg"]
    poly_order = float(pdf_parameters["poly_order"])
    background_path = basic_parameters["background_file"]
    background_enabled = bool(basic_parameters.get("background_enabled", True))

    if background_enabled and background_path and not multiple_graphs:
        bkg_x, bkg_y = extract_data(background_path)

    # 수정된 atom_indices(N=157)를 사용하여 계산 함수 호출
    # Normalization 로직은 ezpit.processing에 없으므로 추가되지 않음.
    # 원소별 조성량(atom_weights)을 직접 넘겨 <f>, <f^2>를 가중 평균으로 계산.
    # 배경은 [bkg_x, bkg_y] 형태로 넘겨, 샘플과 배경의 점 개수나 q 범위가 달라도
    # cal_expSq가 각자의 q축으로 올바르게 보간하도록 한다. (예전에는 bkg_y만
    # 넘겨서 두 파일 길이가 다르면 "fp and xp are not of the same length"
    # 오류가 났다.)
    bkg_arg = [bkg_x, bkg_y] if bkg_y is not None else None
    res = cal_expSq(
        atom_indices,
        scattering_factors,
        exp_data,
        bkg_arg,
        qmin,
        qmax,
        q_step,
        background_scale,
        poly_order,
        True,
        weights=atom_weights,
    )

    if bkg_y is not None:
        return res, [bkg_x, bkg_y * background_scale]
    else:
        return res, [bkg_x, bkg_y]


# ... (이하 함수들은 기존과 동일하게 유지)
def get_fq(Sq, qmin, qmax):
    return cal_fq(qmin, qmax, Sq)


def get_gr(q_range, Sq_or_Fq, control_panel, xyz_data=False, is_Fq=False):
    if xyz_data:
        cal_controls = control_panel.get_cal_parameters()
        rmin = float(cal_controls["rmin"])
        rmax = float(cal_controls["rmax"])
        rstep = float(cal_controls["rstep"])
    else:
        basic_controls = control_panel.get_basic_parameters()
        rmin = float(basic_controls["rmin"])
        rmax = float(basic_controls["rmax"])
        rstep = float(basic_controls["rstep"])

    return cal_expGr_fft(q_range, Sq_or_Fq, rmin, rmax, rstep, is_Fq=is_Fq)


def get_smooth_whittaker(y, control_panel):
    pdf_parameters = control_panel.get_pdf_parameters()
    lambda_fq = float(pdf_parameters["lambda"])
    order = int(pdf_parameters["order"])
    return smooth_whittaker(y, lambda_fq, order)


def get_xyz_graphs(atom_names, atom_positions, control_panel):
    # cal_Sq() looks scattering factors up by *unique-element* index
    # (atom_indices maps each atom to its element in unique_atom_names), so the
    # scattering-factor table must have one row per unique element in that same
    # order. Passing the full per-atom list here would return one row per atom
    # and the unique-index lookup would then read the wrong rows, assigning the
    # wrong form factor to any element whose first appearance in the file does
    # not line up with its unique index (e.g. O and I getting C's form factor).
    unique_atom_names, _, atom_indices = group_atoms(atom_names)
    cal_parameters = control_panel.get_cal_parameters()
    atom_distance_matrix = create_atom_distance_matrix(atom_positions)
    scattering_factors = get_aff_scattering_factors(unique_atom_names)
    qmin = float(cal_parameters["qmin"])
    qmax = float(cal_parameters["qmax"])
    qstep = float(cal_parameters.get("qstep", 0.05))
    q_range, list_Iq, list_Sq, list_Fq, *_ = cal_Sq(
        atom_indices,
        scattering_factors,
        atom_distance_matrix,
        qmin,
        qmax,
        qstep=qstep,
        return_Iq=True,
    )
    return q_range, list_Iq, list_Sq, list_Fq


def get_compton_values(control_panel):
    params = control_panel.get_compton_parameters()
    qmin = float(params["qmin"])
    qmax = float(params["qmax"])
    qstep = float(params["qstep"])
    wavelength = float(params["wavelength"])
    alpha = int(params["alpha"])
    composition_dict = parse_composition(params["composition"])

    # 원소별 조성량을 그대로 가중치로 사용 (소수 조성도 지원).
    unique_atom_names, atom_weights = composition_weights(composition_dict)

    form_factors, atomic_numbers = get_compton_scattering_factors(unique_atom_names)
    compton_parm_only = get_compton_parameter_only()

    # atom_indices는 하위 호환용이며, composition_weights가 주어지면 사용되지 않는다.
    atom_indices = np.arange(len(unique_atom_names))

    return compton_cal_exp(
        atom_indices,
        compton_parm_only,
        form_factors,
        atomic_numbers,
        qmin,
        qmax,
        qstep,
        wavelength,
        alpha,
        weights=atom_weights,
    )
