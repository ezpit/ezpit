# model/processing.py

from ezpit.processing import (cal_expSq, cal_expGr_fft, smooth_whittaker,
                              create_atom_distance_matrix, cal_Sq, compton_cal_exp, cal_fq)
from ezpit.gui.model.elem_data import ElementData
# [수정] convert_atom_names 추가 임포트
from ezpit.gui.model.helpers import parse_composition, group_atoms, extract_data, convert_atom_names
import numpy as np

_elem_data = ElementData()


def get_expSq(exp_data, control_panel, multiple_graphs=False):
    """
    Manipulates the data to get the appropriate S(q) and F(q) data from the experimental data.
    """
    bkg_x = None
    bkg_y = None
    background_scale = 0

    pdf_parameters = control_panel.get_pdf_parameters()
    basic_parameters = control_panel.get_basic_parameters()

    # [NEW] Extract q_step from control panel
    try:
        q_step = float(pdf_parameters.get('qstep', 0.01))
    except (ValueError, TypeError):
        q_step = 0.01

    if q_step <= 0:
        q_step = 0.01

    # -------------------------------------------------------------------------
    # [버그 수정 구간] 원자 개수(N)를 올바르게 계산하도록 수정
    # -------------------------------------------------------------------------
    # 1. 조성 문자열 파싱 -> Dictionary ({'Co':38, 'O':119})
    composition_dict = parse_composition(basic_parameters['composition'])
    # print(f"[DEBUG3] raw composition = {repr(basic_parameters['composition'])}")
    # print(f"[DEBUG3] composition_dict = {dict(composition_dict)}")

    # 2. [핵심 수정] Dictionary를 전체 원자 리스트로 변환 (['Co', ..., 'O'])
    # 기존: atom_names가 dict여서 len(atom_indices)가 2가 되었음.
    # 수정: convert_atom_names를 통해 전체 리스트(157개)로 확장.
    atom_names_list = convert_atom_names(composition_dict)

    # 3. 전체 리스트를 기반으로 인덱스 생성 (len(atom_indices) == 157)
    atom_indices = group_atoms(atom_names_list)[2]

    # 4. 산란 인자는 고유 원소 이름만 있으면 되므로 keys() 사용 (속도 최적화)
    # (helpers.py의 get_aff_scattering_factors는 리스트를 받아 처리하므로 unique list 전달)
    unique_atom_names = list(composition_dict.keys())
    scattering_factors = _elem_data.get_aff_scattering_factors(unique_atom_names)
    # -------------------------------------------------------------------------

    unit_type = basic_parameters['data_format']
    wavelength = float(basic_parameters['wavelength'])

    if unit_type == '2theta':
        theta_rad = np.radians(exp_data[0] / 2)
        exp_data[0] = (4 * np.pi * np.sin(theta_rad)) / wavelength
    elif unit_type == 'q_nmn':
        exp_data[0] = exp_data[0] * 0.1

    qmin = pdf_parameters['qmin']
    qmax = pdf_parameters['qmax']
    background_scale = pdf_parameters['bg']
    poly_order = float(pdf_parameters['poly_order'])
    background_path = basic_parameters['background_file']
    background_enabled = bool(basic_parameters.get('background_enabled', True))

    if background_enabled and background_path and not multiple_graphs:
        bkg_x, bkg_y = extract_data(background_path)

    # 수정된 atom_indices(N=157)를 사용하여 계산 함수 호출
    # Normalization 로직은 ezpit.processing에 없으므로 추가되지 않음.
    res = cal_expSq(
        atom_indices, scattering_factors, exp_data, bkg_y,
        qmin, qmax, q_step, background_scale, poly_order, True
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
        rmin = float(cal_controls['rmin'])
        rmax = float(cal_controls['rmax'])
        rstep = float(cal_controls['rstep'])
    else:
        basic_controls = control_panel.get_basic_parameters()
        rmin = float(basic_controls['rmin'])
        rmax = float(basic_controls['rmax'])
        rstep = float(basic_controls['rstep'])

    return cal_expGr_fft(q_range, Sq_or_Fq, rmin, rmax, rstep, is_Fq=is_Fq)


def get_smooth_whittaker(y, control_panel):
    pdf_parameters = control_panel.get_pdf_parameters()
    lambda_fq = float(pdf_parameters['lambda'])
    order = int(pdf_parameters['order'])
    return smooth_whittaker(y, lambda_fq, order)


def get_xyz_graphs(atom_names, atom_positions, control_panel):
    atom_indices = group_atoms(atom_names)[2]
    cal_parameters = control_panel.get_cal_parameters()
    atom_distance_matrix = create_atom_distance_matrix(atom_positions)
    scattering_factors = _elem_data.get_aff_scattering_factors(atom_names)
    qmin = float(cal_parameters['qmin'])
    qmax = float(cal_parameters['qmax'])
    qstep = float(cal_parameters.get('qstep', 0.05))
    q_range, list_Iq, list_Sq, list_Fq, *_ = cal_Sq(atom_indices, scattering_factors, atom_distance_matrix, qmin, qmax,
                                                    qstep=qstep, return_Iq=True)
    return q_range, list_Iq, list_Sq, list_Fq


def get_compton_values(control_panel):
    params = control_panel.get_compton_parameters()
    qmin = float(params['qmin'])
    qmax = float(params['qmax'])
    qstep = float(params['qstep'])
    wavelength = float(params['wavelength'])
    alpha = int(params['alpha'])
    atom_names = parse_composition(params['composition'])
    _, _, atom_indices = group_atoms(
        atom_names)  # Compton 부분은 group_atoms가 dict를 받아도 되는지 확인 필요하나, helpers.py 구조상 리스트를 넣는게 안전함.
    # Compton은 helpers.py를 따르므로 아래와 같이 수정 권장
    atom_names_list = convert_atom_names(atom_names)
    _, _, atom_indices = group_atoms(atom_names_list)

    form_factors, atomic_numbers = _elem_data.get_compton_scattering_factors(list(atom_names.keys()))  # 고유 원소만 필요
    compton_parm_only = _elem_data.get_compton_parameter_only()
    return compton_cal_exp(atom_indices, compton_parm_only, form_factors, atomic_numbers, qmin, qmax, qstep, wavelength,
                           alpha)

