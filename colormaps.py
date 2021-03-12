"""
Common and custom colormaps for plotting.
"""
import copy
from matplotlib.colors import colorConverter
import matplotlib
import matplotlib as mpl
import matplotlib.cm
import numpy as np


viridis_cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
viridis_cmap.set_bad(color="#AAAAAA")

ylorrd_cmap = copy.copy(matplotlib.cm.get_cmap("YlOrRd"))
ylorrd_cmap.set_bad(color="#AAAAAA")

cdict = {
    "red": [
        (0.0, 1.0, 1.0),
        (0.25, 245 / 255, 245 / 255),
        (0.5, 208 / 255, 208 / 255),
        (1.0, 0.0, 0.0),
    ],
    "green": [
        (0.0, 1.0, 1.0),
        (0.25, 166 / 255, 166 / 255),
        (0.5, 2 / 255, 2 / 255),
        (1.0, 0.0, 0.0),
    ],
    "blue": [
        (0.0, 1.0, 1.0),
        (0.25, 35 / 255, 35 / 255),
        (0.5, 7 / 255, 7 / 255),
        (1.0, 0.0, 0.0),
    ],
}


fall_cmap = matplotlib.colors.LinearSegmentedColormap("fall", cdict, 256)

newcmap2 = mpl.colors.LinearSegmentedColormap.from_list(
    "newcmap2",
    [
        colorConverter.to_rgba(c)
        for c in ["#fff1d7", "#ffda9d", "#ffb362", "#ff8241", "#ff2b29", "#d60026", "#880028",]
    ],
    256,
)
newcmap2._init()
newcmap2.set_bad(color="#AAAAAA")


hnh_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "hnhcmap",
    0.5 * newcmap2(np.linspace(0.0, 1, 256)) + 0.5 * ylorrd_cmap(np.linspace(0.0, 1, 256)),
    256,
)
hnh_cmap.set_bad(color="#AAAAAA")

hnh_cmap_ext = mpl.colors.LinearSegmentedColormap.from_list(
    "hnh_cmap_ext",
    np.vstack(
        [
            np.vstack(
                [
                    np.ones(34),
                    np.concatenate(
                        [np.arange(0.97254902, 1, 0.97254902 - 0.97038062), np.ones(21)]
                    ),
                    np.arange(0.82156863, 1, 0.82156863 - 0.81618608),
                    np.ones(34),
                ]
            ).T[::-1, :][:-1, :],
            hnh_cmap(np.linspace(0.0, 1, 256)),
        ]
    ),
)
hnh_cmap_ext.set_bad(color="#AAAAAA")

hnh_cmap_ext3 = mpl.colors.LinearSegmentedColormap.from_list(
    "hnh_cmap_ext3",
    np.vstack(
        [
            hnh_cmap_ext(np.linspace(0.0, 1, 256)),
            np.vstack(
                [
                    np.arange(0.51764706, 0.15294118, 0.51764706 - 0.52594939),
                    np.zeros(44),
                    np.ones(44) * 0.15294118,
                    np.ones(44),
                ]
            ).T[1:, :],
        ]
    ),
)
hnh_cmap_ext3.set_bad(color="#AAAAAA")

hnh_cmap_ext4 = mpl.colors.LinearSegmentedColormap.from_list(
    "hnh_cmap_ext4", hnh_cmap_ext3(np.linspace(0.0, 1, 512))[16:, :]
)
hnh_cmap_ext4.set_bad(color="#AAAAAA")

hnh_cmap_ext5 = mpl.colors.LinearSegmentedColormap.from_list(
    "hnh_cmap_ext5", hnh_cmap_ext3(np.linspace(0.0, 1, 512))[32:, :]
)
hnh_cmap_ext5.set_bad(color="#AAAAAA")

color1 = colorConverter.to_rgba("white")
color2 = colorConverter.to_rgba("black")

bwcmap = mpl.colors.LinearSegmentedColormap.from_list("bwcmap", [color1, color2], 256)
bwcmap._init()
alphas = np.linspace(0, 0.2, bwcmap.N + 3)
bwcmap._lut[:, -1] = alphas
