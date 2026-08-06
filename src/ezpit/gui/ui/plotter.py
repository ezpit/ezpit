# plotter.py

import pyqtgraph as pg

def make_pen(color_tuple, width=2):
    return pg.mkPen(color=color_tuple, width=width)

def setup_plot(plot_widget, x_label, y_label):
    plot_widget.clear()
    # ⬇⬇⬇ 이 두 줄이 삭제되었는지 확인하세요 (필수) ⬇⬇⬇
    # plot_widget.setLabel('bottom', x_label, **{'color': '#000'})
    # plot_widget.setLabel('left', y_label, **{'color': '#000'})
    # ⬆⬆⬆ 여기까지 ⬆⬆⬆
    plot_widget.showGrid(x=False, y=False, alpha=0.3)

def plot_curve(plot_widget, x, y, pen, visible=True):
    if x is not None and y is not None:
        curve = plot_widget.plot(x, y, pen=pen)
        curve.setVisible(visible)
        return curve
    return None

def update_visibility(layout, checkboxes, plots, parent_widget):
    for i, plot in enumerate(plots):
        visible = checkboxes[i].isChecked()
        plot.setVisible(visible)
        layout.setStretch(layout.indexOf(plot), 1 if visible else 0)
    parent_widget.layout().invalidate()
    parent_widget.updateGeometry()
