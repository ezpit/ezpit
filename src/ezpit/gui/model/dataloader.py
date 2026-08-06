# model/dataloader.py
from ezpit.gui.model.processing import get_expSq, get_gr, get_smooth_whittaker, get_xyz_graphs, get_fq
from ezpit.gui.model.helpers import extract_data, load_atom_name_positions, group_atoms, get_q_range, trim_and_pad
from ezpit.gui.model.saver import write_config_file
import os
import json
from ezpit.gui.model.extensions import run_transforms
from PySide6.QtWidgets import QMessageBox

def get_graph_data(path, extension, control_panel, multiple_graphs=False):
    if extension in ['.iq', '.chi', '.xy', '.dat', '.txt', '.sq', '.fq', '.gr', '.caliq', '.calfq', '.calsq', '.calgr', '.compton']:
        x_raw_input, y_raw_input = extract_data(path)
        x, y = x_raw_input, y_raw_input

        xs = [[], [], [], []]
        ys = [[], [], [], []]
        bkg_x = None
        bkg_y = None
        list_Sq = None
        Fq_smoothed = None
        mean_sq_fi = None
        sq_mean_fi = None
        r_smoothed = None
        G_smoothed = None

        param_source = control_panel.get_pdf_parameters()
        qmin = float(param_source['qmin'])
        qmax = float(param_source['qmax'])

        if extension in ['.chi', '.iq', '.xy', '.dat', '.txt', '.caliq']:
            try:
                res = get_expSq([x, y], control_panel, multiple_graphs=multiple_graphs)
                q_range, list_Iq, scaled_expIq, list_scaled_bkgIq, list_Sq_unnorm, norm_list_Sq, list_Fq, mean_sq_fi, sq_mean_fi = \
                res[0]
                bkg_x = res[1][0]
                bkg_y = res[1][1]

                r_orig, G_orig = get_gr(q_range, norm_list_Sq, control_panel)

                Fq_smoothed = get_smooth_whittaker(list_Fq, control_panel)
                r_smoothed, G_smoothed = get_gr(q_range, Fq_smoothed, control_panel, is_Fq=True)

                xs = [q_range, q_range, q_range, r_orig]
                ys = [list_Iq, norm_list_Sq, list_Fq, G_orig]
                list_Sq = list_Sq_unnorm

                ctx = {
                    "q_range": q_range,
                    "Fq": list_Fq,
                    "Fq_smoothed": Fq_smoothed,
                    "params": control_panel,
                    "get_gr_fn": get_gr
                }
                extension_data = run_transforms("post_fq", ctx)

            except Exception as e:
                msg = QMessageBox(control_panel)
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Error Processing File")
                msg.setText("Failed to compute structure function or G(r).")
                msg.setInformativeText(str(e))
                msg.exec()
                return xs, ys, bkg_x, bkg_y, x_raw_input, y_raw_input, None, None, None, None, None, None

        elif extension in ['.gr', '.calgr']:
            xs[3] = x
            ys[3] = y

        elif extension in ['.fq', '.calfq']:
            xs[2], ys[2] = x, y

            Fq_smoothed = get_smooth_whittaker(y, control_panel)

            q_range = get_q_range(qmin, qmax, len(xs[2]))

            control_panel.set_pdf_parameters({
                "qmin": min(x),
                "qmax": max(x)
            })

            r, G = get_gr(q_range, ys[2], control_panel, is_Fq=True)
            r_smoothed, G_smoothed = get_gr(q_range, Fq_smoothed, control_panel, is_Fq=True)

            xs[3] = r
            ys[3] = G

        elif extension in ['.sq', '.calsq']:
            xs[1] = x
            ys[1] = y

            control_panel.set_pdf_parameters({
                "qmin": min(x),
                "qmax": max(x)
            })

            xs[2] = x
            ys[2], q_range = get_fq(y, min(x), max(x))

            Fq_smoothed = get_smooth_whittaker(ys[2], control_panel)

            r_smoothed, G_smoothed = get_gr(q_range, Fq_smoothed, control_panel, is_Fq=True)
            r, G = get_gr(q_range, ys[1], control_panel)

            xs[3] = r
            ys[3] = G
        elif extension == '.compton':
            xs[0] = x
            ys[0] = y

        if extension not in ['.chi', '.iq', '.xy', '.dat', '.txt', '.caliq']:
            x = x_raw_input
            y = y_raw_input
        else:
            x, y = x_raw_input, y_raw_input

        return xs, ys, bkg_x, bkg_y, x, y, list_Sq, Fq_smoothed, mean_sq_fi, sq_mean_fi, r_smoothed, G_smoothed
    else:
        atom_names, atom_positions = load_atom_name_positions(path)
        q_range, list_Iq, list_Sq, list_Fq = get_xyz_graphs(atom_names, atom_positions, control_panel)
        r, G = get_gr(q_range, list_Sq, control_panel, xyz_data=True)

        xs = [q_range, q_range, q_range, r]
        ys = [list_Iq, list_Sq, list_Fq, G]

        return xs, ys, None, None, None, None, list_Sq, None, None, None, None, None


def get_xyz_data(path):
    return load_atom_name_positions(path)


def determine_config_type(params: dict) -> str:
    if "composition" in params and "wavelength" in params:
        return "basic"
    elif "poly_order" in params and "lambda" in params:
        return "pdf"
    elif "cal_rmin" in params or "rmin" in params and "cal_qmin" in params:
        return "cal"
    else:
        return "unknown"


def get_config_data(path):
    _, ext = os.path.splitext(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    config_path = os.path.join(os.path.dirname(path), f"{base_name}.ezpdf_config.json")

    # [수정] pdf_controls에 "qstep" 기본값 추가
    default_config = {
        "basic_controls": {
            "data_format": "2theta",
            "source_type": "Custom",
            "wavelength": "0.1",
            "background_file": None,
            "composition": "C 1 O 2",
            "rmin": "0",
            "rmax": "30",
            "rstep": "0.01"
        },
        "pdf_controls": {
            "bg": 1.0,
            "qmax": 24.0,
            "qmin": 0.1,
            "poly_order": 9.0,
            "lambda": "1000",
            "order": "2",
            "qstep": 0.01  # <-- [추가됨]
        },
        "compton_controls": {
            "wavelength": "0.1665",
            "alpha": 3,
            "composition": "",
            "qmin": 0.0,
            "qmax": 30.0,
            "qstep": 0.01
        }
    }

    if ext == ".xyz":
        default_config["cal_controls"] = {
            "qmax": 30.0,
            "qmin": 0.0,
            "rmin": "0",
            "rmax": "30",
            "rstep": "0.01"
        }

    if not os.path.exists(config_path):
        write_config_file(default_config, config_path)
        return default_config

    with open(config_path, "r") as f:
        config = json.load(f)

    updated = False

    for section, defaults in default_config.items():
        if section not in config:
            config[section] = defaults
            updated = True
        else:
            for key, value in defaults.items():
                if key not in config[section]:
                    config[section][key] = value
                    updated = True

    if updated:
        write_config_file(config, config_path)

    return config