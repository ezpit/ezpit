from ezpit.gui.model.optimizer import optimize_background_scale, optimize_qmax, optimize_poly_order
import numpy as np

def apply_optimal_pdf_settings(control_panel, xs, ys, bkg_x, bkg_y, raw_x, raw_y, list_Sq):
    """
    Sets optimal PDF parameters based on I(Q), F(Q), and S(Q) data.
    Applies updates after each optimization step to reflect dependent changes.
    """
    try:
        iq_x, fq_x, sq_x, _ = xs
        iq_y, fq_y, sq_y, _ = ys

        # Initialize params with default values
        params = {
            "bg": 0.0,
            "qmin": 0.1,
            "qmax": max(raw_x),
            "poly_order": 11.0,               #############11
            "order_enabled": False,
            "lambda": "1000",
            "order": "2",
        }

        # set the control panel to these default values before calculating poly order/qmax for consistency
        control_panel.set_pdf_parameters(params)


        #  Optimize poly order (depends on qmax)
        params["poly_order"] = optimize_poly_order(params["qmax"], sq_x, list_Sq)
        control_panel.set_pdf_parameters(params)

        # Optimize qmax
        #params["qmax"] = optimize_qmax(fq_x, fq_y)
        #control_panel.set_pdf_parameters(params)

    except Exception as e:
        print(f"Warning: Failed to apply optimal PDF settings: {e}")
