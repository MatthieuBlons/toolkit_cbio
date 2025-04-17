import numpy as np
from slide.utils import get_x_y_to
import pandas as pd


def get_feat(row, feat, dim_0=None, dim_l=None):
    d = {
        "x": None,
        "y": None,
        "w": None,
        "h": None,
        feat: None,
    }
    xmin = row["x"]
    ymin = row["y"]
    w = row["w"]
    h = row["h"]
    if dim_0 and dim_l:
        xmax, ymax = xmin + w, ymin + h
        xmin, ymin = get_x_y_to((xmin, ymin), dim_0, dim_l)
        xmax, ymax = get_x_y_to((xmax, ymax), dim_0, dim_l)
        w, h = (
            xmax - xmin,
            ymax - ymin,
        )
    d["x"] = xmin
    d["y"] = ymin
    d["w"] = w
    d["h"] = h
    d[feat] = row[feat]
    return pd.Series(d)


def get_area(x, y, w, h):
    xmin, xmax = x, x + w
    ymin, ymax = y, y + h
    xvect = np.array([x for x in range(int(xmin), int(xmax))])
    yvect = np.array([y for y in range(int(ymin), int(ymax))])
    area = np.meshgrid(xvect, yvect)
    return area, np.ravel(area[0]).tolist(), np.ravel(area[1]).tolist()
