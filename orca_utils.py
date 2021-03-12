"""
This module contains the utilities for Orca-based applications,
including a class for structural variants and plotting utilities.
"""
import os
import pathlib
from copy import deepcopy
from collections import OrderedDict, namedtuple
from bisect import bisect

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from colormaps import hnh_cmap_ext5, bwcmap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ORCA_PATH = str(pathlib.Path(__file__).parent.absolute())


def _draw_region(ax, linestart, lineend, color, matlen):
    ax.plot(
        [-matlen / 50, -matlen / 50],
        [matlen * linestart - 0.5 - 0.1, (matlen) * lineend - 0.5 + 0.1],
        solid_capstyle="butt",
        color=color,
        linewidth=8,
        zorder=10,
        clip_on=False,
    )


def _draw_site(ax, linepos, mode, matlen, color="black"):
    if mode == "double":
        ax.plot(
            [-(matlen) / 20, -0.5],
            [(matlen) * linepos - (matlen) / 100.0 - 0.5, (matlen) * linepos - 0.5],
            color=color,
            linewidth=0.2,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [-(matlen) / 20, -0.5],
            [(matlen) * linepos + (matlen) / 100.0 - 0.5, (matlen) * linepos - 0.5],
            color=color,
            linewidth=0.2,
            zorder=10,
            clip_on=False,
        )
    elif mode == "single":
        ax.plot(
            [-(matlen) / 20, -0.5],
            [(matlen) * linepos - 0.5, (matlen) * linepos - 0.5],
            color=color,
            linewidth=0.2,
            zorder=10,
            clip_on=False,
        )


def genomeplot(
    output,
    show_genes=False,
    show_tracks=False,
    show_coordinates=True,
    unscaled=False,
    file=None,
    cmap=None,
    unscaled_cmap=None,
    colorbar=True,
    maskpred=False,
    vmin=-1,
    vmax=2,
):
    """
    Plot the multiscale prediction outputs for 32Mb output.

    Parameters
    ----------
    output : dict
        The result dictionary to plot as returned by `genomepredict_256Mb`.
    show_genes : bool, optional
        Default is False. If True, plot the retrieved
        gene annotations corresponding to all windows used for the multiscale prediction.
    show_tracks : bool, optional
        Default is False. If True, plot the retrieved
        chromatin tracks for CTCF, chromatin accessibility and histone marks 
        for all windows used for the multiscale prediction. 
    show_coordinates : bool, optional
        Default is True. If True, annotate the generated plot with the 
        genome coordinates.
    unscaled : bool, optional
        Default is False. If True, plot the predictions and observations
        without normalizing by distance-based expectation.
    file : str or None, optional
        Default is None. The output file prefix. No output file is generated
        if set to None.
    cmap : str or None, optional
        Default is None. The colormap for plotting scaled interactions (log
        fold over distance-based background). If None, use colormaps.hnh_cmap_ext5.
    unscaled_cmap : str or None, optional
        Default is None. The colormap for plotting unscaled interactions (log
        balanced contact score). If None, use colormaps.hnh_cmap_ext5.
    colorbar : bool, optional
        Default is True. Whether to plot the colorbar.
    maskpred : bool, optional
        Default is True. If True, the prediction heatmaps are masked at positions
        where the observed data have missing values when observed data are provided
        in output dict.
    vmin : int, optional
        Default is -1. The lowerbound value for heatmap colormap.
    vmax : int, optional
        Default is 2. The upperbound value for heatmap colormap.

    Returns
    -------
    None
    """
    if cmap is None:
        cmap = hnh_cmap_ext5
    if unscaled_cmap is None:
        unscaled_cmap = hnh_cmap_ext5

    n_axes = 2 + 2 * (output["experiments"] is not None)

    fig, all_axes = plt.subplots(figsize=(36, 6 * n_axes), nrows=n_axes, ncols=6)

    for row_axes in all_axes:
        for ax in row_axes:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    for i, xlabel in enumerate(["1Mb", "2Mb", "4Mb", "8Mb", "16Mb", "32Mb"]):
        all_axes[-1, i].set_xlabel(xlabel, labelpad=20, fontsize=20, weight="black")

    if output["experiments"] is not None:
        all_axes[0, 0].set_ylabel(
            "H1-ESC Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[1, 0].set_ylabel(
            "H1-ESC Obs",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[2, 0].set_ylabel(
            "HFF Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[3, 0].set_ylabel(
            "HFF Obs",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
    else:
        all_axes[0, 0].set_ylabel(
            "H1-ESC Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[1, 0].set_ylabel(
            "HFF Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )

    current_row = 0
    for ii, ax in enumerate(reversed(all_axes[current_row])):
        s = int(output["start_coords"][ii])
        e = int(output["end_coords"][ii])
        regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
        if show_coordinates:
            ax.set_title(regionstr, fontsize=14, pad=4)
        if unscaled:
            plotmat = output["predictions"][0][ii] + np.log(output["normmats"][0][ii])
            im = ax.imshow(
                plotmat,
                interpolation="none",
                cmap=unscaled_cmap,
                vmax=np.max(np.diag(plotmat, k=1)),
            )
        else:
            plotmat = output["predictions"][0][ii]
            im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)

        if output["annos"]:
            for r in output["annos"][ii]:
                if len(r) == 3:
                    _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                elif len(r) == 2:
                    _draw_site(ax, r[0], r[1], plotmat.shape[1])
            ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
            ax.invert_yaxis()

        if maskpred:
            if output["experiments"]:
                ax.imshow(
                    np.isnan(output["experiments"][0][ii]), interpolation="none", cmap=bwcmap,
                )

    current_row += 1

    if output["experiments"]:
        for ii, ax in enumerate(reversed(all_axes[current_row])):
            s = int(output["start_coords"][ii])
            e = int(output["end_coords"][ii])
            regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
            if show_coordinates:
                ax.set_title(regionstr, fontsize=14, pad=4)
            if unscaled:
                plotmat = output["experiments"][0][ii] + np.log(output["normmats"][0][ii])
                im = ax.imshow(
                    plotmat,
                    interpolation="none",
                    cmap=unscaled_cmap,
                    vmax=np.max(np.diag(plotmat, k=1)),
                )
            else:
                plotmat = output["experiments"][0][ii]
                im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
            if output["annos"]:
                for r in output["annos"][ii]:
                    if len(r) == 3:
                        _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                    elif len(r) == 2:
                        _draw_site(ax, r[0], r[1], plotmat.shape[1])
                ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
                ax.invert_yaxis()
        current_row += 1

    for ii, ax in enumerate(reversed(all_axes[current_row])):
        s = int(output["start_coords"][ii])
        e = int(output["end_coords"][ii])
        regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
        if show_coordinates:
            ax.set_title(regionstr, fontsize=14, pad=4)
        if unscaled:
            plotmat = output["predictions"][1][ii] + np.log(output["normmats"][1][ii])
            im = ax.imshow(
                plotmat,
                interpolation="none",
                cmap=unscaled_cmap,
                vmax=np.max(np.diag(plotmat, k=1)),
            )
        else:
            plotmat = output["predictions"][1][ii]
            im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)

        if output["annos"]:
            for r in output["annos"][ii]:
                if len(r) == 3:
                    _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                elif len(r) == 2:
                    _draw_site(ax, r[0], r[1], plotmat.shape[1])
            ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
            ax.invert_yaxis()

        if maskpred:
            if output["experiments"]:
                ax.imshow(
                    np.isnan(output["experiments"][1][ii]), interpolation="none", cmap=bwcmap,
                )
    current_row += 1

    if output["experiments"]:
        for ii, ax in enumerate(reversed(all_axes[current_row])):
            s = int(output["start_coords"][ii])
            e = int(output["end_coords"][ii])
            regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
            if show_coordinates:
                ax.set_title(regionstr, fontsize=14, pad=4)
            if unscaled:
                plotmat = output["experiments"][1][ii] + np.log(output["normmats"][1][ii])
                im = ax.imshow(
                    plotmat,
                    interpolation="none",
                    cmap=unscaled_cmap,
                    vmax=np.max(np.diag(plotmat, k=1)),
                )
            else:
                plotmat = output["experiments"][1][ii]
                im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
            if output["annos"]:
                for r in output["annos"][ii]:
                    if len(r) == 3:
                        _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                    elif len(r) == 2:
                        _draw_site(ax, r[0], r[1], plotmat.shape[1])
                ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
                ax.invert_yaxis()
        current_row += 1

    if colorbar:
        fig.colorbar(im, ax=all_axes.ravel().tolist(), fraction=0.02, shrink=0.1, pad=0.005)

    if show_genes:
        for p in [
            ORCA_PATH + "/resources/hg38.refGeneSelectMANE.bed.gz",
            ORCA_PATH + "/resources/hg38.refGeneSelectMANE.bed.gz.tbi",
        ]:
            if not os.path.exists(p):
                show_genes = False
                print(
                    "`show_genes` is turned off because resource file " + p + " is not available."
                )
                break
    if show_tracks:
        for p in [
            ORCA_PATH + "/extra/H1_CTCF_ENCFF473IZV.bigWig",
            ORCA_PATH + "/extra/H1_RAD21_ENCFF913JGA.bigWig",
            ORCA_PATH + "/extra/H1_DNase_ENCFF131HMO.bigWig",
            ORCA_PATH + "/extra/H1_H3K4me3_ENCFF623ZAW.bigWig",
            ORCA_PATH + "/extra/H1_POLR2A_ENCFF379IRQ.bigWig",
            ORCA_PATH + "/extra/H1_H3K27ac_ENCFF423TVA.bigWig",
            ORCA_PATH + "/extra/H1_H3K4me1_ENCFF584AVI.bigWig",
            ORCA_PATH + "/extra/H1_H3K36me3_ENCFF141YAA.bigWig",
            ORCA_PATH + "/extra/H1_H3K27me3_ENCFF912ZUR.bigWig",
            ORCA_PATH + "/extra/H1_H3K9me3_ENCFF752UGN.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_CTCF_ENCFF761RHS.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_DNase_ENCFF113YFF.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_H3K4me3_ENCFF442WNT.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_H3K27ac_ENCFF078JZB.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_H3K4me1_ENCFF449DEA.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_H3K36me3_ENCFF954UKB.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_H3K27me3_ENCFF027GWJ.bigWig",
            ORCA_PATH + "/extra/foreskin_fibroblast_H3K9me3_ENCFF946TXL.bigWig",
        ]:
            if not os.path.exists(p):
                show_tracks = False
                print(
                    "`show_tracks` is turned off because resource file " + p + " is not available."
                )
                break
    if show_genes or show_tracks:
        browser_tracks = (
            """
        [spacer]
        height = 0.5
        [x-axis]
        where = top
        fontsize = 12
        [spacer]
        height = 0.05
        """
            + (
                """
        [test gtf collapsed]
        file = {ORCA_PATH}/resources/hg38.refGeneSelectMANE.bed.gz
        height = 25
        merge_transcripts = true
        prefered_name = gene_name
        max_labels = 10000
        fontsize = 9
        file_type = bed
        gene_rows = 40
        display = stacked
        """.format(
                    ORCA_PATH=ORCA_PATH
                )
                if show_genes
                else ""
            )
            + (
                """
        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_CTCF_ENCFF473IZV.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-CTCF
        summary_method = mean
        file_type = bigwig

        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_RAD21_ENCFF913JGA.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-RAD21
        summary_method = mean
        file_type = bigwig


        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_DNase_ENCFF131HMO.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-DNase
        summary_method = mean
        file_type = bigwig
        color = #2A6D8F


        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_H3K4me3_ENCFF623ZAW.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-H3K4me3
        summary_method = mean
        file_type = bigwig
        color = #E76F51
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_POLR2A_ENCFF379IRQ.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-POL2
        summary_method = mean
        file_type = bigwig
        color = #E76F51

        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_H3K27ac_ENCFF423TVA.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-H3K27ac
        summary_method = mean
        file_type = bigwig
        color = #F4A261
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_H3K4me1_ENCFF584AVI.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-H3K4me1
        summary_method = mean
        file_type = bigwig
        color = #F4A261
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_H3K36me3_ENCFF141YAA.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-H3K36me3
        summary_method = mean
        file_type = bigwig
        color = #E9C46A
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_H3K27me3_ENCFF912ZUR.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-H3K27me3
        summary_method = mean
        file_type = bigwig
        color = #264653

        
        [bigwig file test]
        file = {ORCA_PATH}/extra/H1_H3K9me3_ENCFF752UGN.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = H1-H3K9me3
        summary_method = mean
        file_type = bigwig
        color = #264653
        
        [spacer]
        height = 2
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_CTCF_ENCFF761RHS.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-CTCF
        summary_method = mean
        file_type = bigwig


        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_DNase_ENCFF113YFF.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-DNase
        summary_method = mean
        file_type = bigwig
        color = #2A6D8F


        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_H3K4me3_ENCFF442WNT.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-H3K4me3
        summary_method = mean
        file_type = bigwig
        color = #E76F51


        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_H3K27ac_ENCFF078JZB.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-H3K27ac
        summary_method = mean
        file_type = bigwig
        color = #F4A261
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_H3K4me1_ENCFF449DEA.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-H3K4me1
        summary_method = mean
        file_type = bigwig
        color = #F4A261
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_H3K36me3_ENCFF954UKB.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-H3K36me3
        summary_method = mean
        file_type = bigwig
        color = #E9C46A
        
        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_H3K27me3_ENCFF027GWJ.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-H3K27me3
        summary_method = mean
        file_type = bigwig
        color = #264653

        
        [bigwig file test]
        file = {ORCA_PATH}/extra/foreskin_fibroblast_H3K9me3_ENCFF946TXL.bigWig
        # height of the track in cm (optional value)
        height = 2
        title = HFF-H3K9me3
        summary_method = mean
        file_type = bigwig
        color = #264653
    
        """.format(
                    ORCA_PATH=ORCA_PATH
                )
                if show_tracks
                else ""
            )
        )

        with open("/dev/shm/temp.ini", "w") as fh:
            fh.write(browser_tracks)

        gbfigs = []
        for ii, label in enumerate(["32Mb", "16Mb", "8Mb", "4Mb", "2Mb", "1Mb"]):
            regionstr = (
                output["chr"]
                + ":"
                + str(int(output["start_coords"][ii]))
                + "-"
                + str(int(output["end_coords"][ii]))
            )

            import pygenometracks.plotTracks
            import uuid

            filename = str(uuid.uuid4())
            args = (
                f"--tracks /dev/shm/temp.ini --region {regionstr} "
                "--trackLabelFraction 0.03 --width 40 --dpi 10 "
                f"--outFileName /dev/shm/{filename}.png --title {label}".split()
            )
            _ = pygenometracks.plotTracks.main(args)
            gbfigs.append(plt.gcf())

            os.remove(f"/dev/shm/{filename}.png")

        if file is not None:
            with PdfPages(file) as pdf:
                pdf.savefig(fig, dpi=300)
                plt.show()
            with PdfPages((".").join(file.split(".")[:-1]) + ".anno.pdf") as pdf:
                for fig in reversed(gbfigs):
                    pdf.savefig(fig)
    else:
        if file is not None:
            with PdfPages(file) as pdf:
                pdf.savefig(fig, dpi=300)
    plt.close("all")


def genomeplot_256Mb(
    output,
    show_coordinates=True,
    unscaled=False,
    file=None,
    cmap=None,
    unscaled_cmap=None,
    colorbar=True,
    maskpred=True,
    vmin=-1,
    vmax=2,
):
    """
    Plot the multiscale prediction outputs for 256Mb output.

    Parameters
    ----------
    output : dict
        The result dictionary to plot as returned by `genomepredict_256Mb`.
    show_coordinates : bool, optional
        Default is True. If True, annotate the generated plot with the 
        genome coordinates.
    unscaled : bool, optional
        Default is False. If True, plot the predictions and observations
        without normalizing by distance-based expectation.
    file : str or None, optional
        Default is None. The output file prefix. No output file is generated
        if set to None.
    cmap : str or None, optional
        Default is None. The colormap for plotting scaled interactions (log
        fold over distance-based background). If None, use colormaps.hnh_cmap_ext5.
    unscaled_cmap : str or None, optional
        Default is None. The colormap for plotting unscaled interactions (log
        balanced contact score). If None, use colormaps.hnh_cmap_ext5.
    colorbar : bool, optional
        Default is True. Whether to plot the colorbar.
    maskpred : bool, optional
        Default is True. If True, the prediction heatmaps are masked at positions
        where the observed data have missing values when observed data are provided
        in output dict.
    vmin : int, optional
        Default is -1. The lowerbound value for heatmap colormap.
    vmax : int, optional
        Default is 2. The upperbound value for heatmap colormap.

    Returns
    -------
    None
    """
    if cmap is None:
        cmap = hnh_cmap_ext5
    if unscaled_cmap is None:
        unscaled_cmap = hnh_cmap_ext5

    n_axes = 2 + 2 * (output["experiments"] is not None)

    fig, all_axes = plt.subplots(figsize=(24, 6 * n_axes), nrows=n_axes, ncols=4)

    for row_axes in all_axes:
        for ax in row_axes:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    for i, xlabel in enumerate(["32Mb", "64Mb", "128Mb", "256Mb"]):
        all_axes[-1, i].set_xlabel(xlabel, labelpad=20, fontsize=20, weight="black")

    if output["experiments"] is not None:
        all_axes[0, 0].set_ylabel(
            "H1-ESC Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[1, 0].set_ylabel(
            "H1-ESC Obs",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[2, 0].set_ylabel(
            "HFF Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[3, 0].set_ylabel(
            "HFF Obs",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
    else:
        all_axes[0, 0].set_ylabel(
            "H1-ESC Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )
        all_axes[1, 0].set_ylabel(
            "HFF Pred",
            labelpad=20,
            fontsize=20,
            weight="black",
            rotation="horizontal",
            ha="right",
            va="center",
        )

    current_row = 0
    for ii, ax in enumerate(reversed(all_axes[current_row])):
        s = int(output["start_coords"][ii])
        e = int(output["end_coords"][ii])
        padlen = int(output["start_coords"][ii] + 256000000 / 2 ** (ii)) - e
        if padlen > 0 and output["padding_chr"] is not None:
            regionstr = (
                output["chr"]
                + ":"
                + str(s)
                + "-"
                + str(e)
                + "; "
                + output["padding_chr"]
                + ":0-"
                + str(padlen)
            )
        else:
            regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
        if show_coordinates:
            ax.set_title(regionstr, fontsize=14, pad=4)
        if unscaled:
            plotmat = output["predictions"][0][ii] + np.log(output["normmats"][0][ii])
            im = ax.imshow(
                plotmat,
                interpolation="none",
                cmap=unscaled_cmap,
                vmax=np.max(np.diag(plotmat, k=1)),
            )
        else:
            plotmat = output["predictions"][0][ii]
            im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
        if padlen > 0:
            # draw chr boundary
            chrlen_ratio = 1 - padlen / (256000000 / 2 ** (ii))
            ax.plot(
                [chrlen_ratio * plotmat.shape[0] - 0.5, chrlen_ratio * plotmat.shape[0] - 0.5],
                [-0.5, plotmat.shape[0] - 0.5],
                color="black",
                linewidth=0.2,
                zorder=10,
            )
            ax.plot(
                [-0.5, plotmat.shape[0] - 0.5],
                [chrlen_ratio * plotmat.shape[0] - 0.5, chrlen_ratio * plotmat.shape[0] - 0.5],
                color="black",
                linewidth=0.2,
                zorder=10,
            )
        if output["annos"]:
            for r in output["annos"][ii]:
                if len(r) == 3:
                    _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                elif len(r) == 2:
                    _draw_site(ax, r[0], r[1], plotmat.shape[1])
            ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
            ax.invert_yaxis()

        if maskpred:
            if output["experiments"]:
                ax.imshow(
                    np.isnan(output["experiments"][0][ii]), interpolation="none", cmap=bwcmap,
                )

    current_row += 1

    if output["experiments"]:
        for ii, ax in enumerate(reversed(all_axes[current_row])):
            s = int(output["start_coords"][ii])
            e = int(output["end_coords"][ii])
            padlen = int(output["start_coords"][ii] + 256000000 / 2 ** (ii)) - e
            if padlen > 0 and output["padding_chr"] is not None:
                regionstr = (
                    output["chr"]
                    + ":"
                    + str(s)
                    + "-"
                    + str(e)
                    + "; "
                    + output["padding_chr"]
                    + ":0-"
                    + str(padlen)
                )
            else:
                regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
            if show_coordinates:
                ax.set_title(regionstr, fontsize=14, pad=4)
            if unscaled:
                plotmat = output["experiments"][0][ii] + np.log(output["normmats"][0][ii])
                im = ax.imshow(
                    plotmat,
                    interpolation="none",
                    cmap=unscaled_cmap,
                    vmax=np.max(np.diag(plotmat, k=1)),
                )
            else:
                plotmat = output["experiments"][0][ii]
                im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
            if padlen > 0:
                # draw chr boundary
                chrlen_ratio = 1 - padlen / (256000000 / 2 ** (ii))
                ax.plot(
                    [chrlen_ratio * 250 + 0.5, chrlen_ratio * 250 + 0.5],
                    [-0.5, 250.5],
                    color="black",
                    linewidth=0.2,
                    zorder=10,
                )
                ax.plot(
                    [-0.5, 250.5],
                    [chrlen_ratio * 250 + 0.5, chrlen_ratio * 250 + 0.5],
                    color="black",
                    linewidth=0.2,
                    zorder=10,
                )
            if output["annos"]:
                for r in output["annos"][ii]:
                    if len(r) == 3:
                        _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                    elif len(r) == 2:
                        _draw_site(ax, r[0], r[1], plotmat.shape[1])
                ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
                ax.invert_yaxis()
        current_row += 1

    for ii, ax in enumerate(reversed(all_axes[current_row])):
        s = int(output["start_coords"][ii])
        e = int(output["end_coords"][ii])
        regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
        if show_coordinates:
            ax.set_title(regionstr, fontsize=14, pad=4)
        if unscaled:
            plotmat = output["predictions"][1][ii] + np.log(output["normmats"][1][ii])
            im = ax.imshow(
                plotmat,
                interpolation="none",
                cmap=unscaled_cmap,
                vmax=np.max(np.diag(plotmat, k=1)),
            )
        else:
            plotmat = output["predictions"][1][ii]
            im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)

        if output["annos"]:
            for r in output["annos"][ii]:
                if len(r) == 3:
                    _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                elif len(r) == 2:
                    _draw_site(ax, r[0], r[1], plotmat.shape[1])
            ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
            ax.invert_yaxis()

        if maskpred:
            if output["experiments"]:
                ax.imshow(
                    np.isnan(output["experiments"][1][ii]), interpolation="none", cmap=bwcmap,
                )
    current_row += 1

    if output["experiments"]:
        for ii, ax in enumerate(reversed(all_axes[current_row])):
            s = int(output["start_coords"][ii])
            e = int(output["end_coords"][ii])
            regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
            if show_coordinates:
                ax.set_title(regionstr, fontsize=14, pad=4)
            if unscaled:
                plotmat = output["experiments"][1][ii] + np.log(output["normmats"][1][ii])
                im = ax.imshow(
                    plotmat,
                    interpolation="none",
                    cmap=unscaled_cmap,
                    vmax=np.max(np.diag(plotmat, k=1)),
                )
            else:
                plotmat = output["experiments"][1][ii]
                im = ax.imshow(plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
            if output["annos"]:
                for r in output["annos"][ii]:
                    if len(r) == 3:
                        _draw_region(ax, r[0], r[1], r[2], plotmat.shape[1])
                    elif len(r) == 2:
                        _draw_site(ax, r[0], r[1], plotmat.shape[1])
                ax.axis([-0.5, plotmat.shape[1] - 0.5, -0.5, plotmat.shape[1] - 0.5])
                ax.invert_yaxis()
        current_row += 1

    if colorbar:
        fig.colorbar(im, ax=all_axes.ravel().tolist(), fraction=0.02, shrink=0.1, pad=0.005)

    if file is not None:
        with PdfPages(file) as pdf:
            pdf.savefig(fig, dpi=300)
    plt.close("all")


GRange = namedtuple("GRange", ["chr", "start", "end", "strand"])
LGRange = namedtuple("LGRange", ["len", "ref"])


class StructuralChange2(object):
    """
    This class stores and manupulating structural changes for a single 
    chromosome and allow querying the mutated chromosome by coordinates by providing 
    utilities for retrieving the corresponding reference genome segments.

    The basic operations that StructuralChange2 supports are duplication, deletion, 
    inversion, insertion, and concatenation. StructuralChange2 objects can be concatenated with '+' 
    operator, this operation allows concatenating two chromosomes. '+' can be combined with
    other basic operations to create fused chromosomes.

    These operations can be used sequentially to introduce arbitrarily complex 
    structural changes. However, note that the coordinates are dynamically updated 
    after each operation reflecting the current state of the chromosome, thus coordinates 
    specified in later operation must take into account of the effects of all previous
    operations.

    Parameters
    ----------
    chr_name : str
        Name of the reference chromosome.
    length : int
        The length of the reference chromosome.


    Attributes
    -------
    segments : list(LGRange)
        List of reference genome segments that constitute the (mutated)
        chromosome. Each element is a LGRange namedtuple (length and a 
        GRange namedtuple (chr: str, start: int, end: int, strand: str)).
    chr_name : str
        Name of the chromosome
    coord_points : list(int)
        Stores `N+1` key coordinates where `N` is the number of segments. The
        key coordinates are 0, segment junction positions, and chromosome end 
        coordinate. `coord_points` reflects the current state of the chromosome.
    """

    def __init__(self, chr_name, length):
        self.segments = [LGRange(length, GRange(chr_name, 0, length, "+"))]
        self.chr_name = chr_name
        self.coord_points = [0, length]

    def coord_sync(self):
        self.coord_points = [0]
        for seg in self.segments:
            self.coord_points.append(self.coord_points[-1] + seg.len)

    def _split(self, pos):
        segind = bisect(self.coord_points, pos) - 1
        segstart = self.coord_points[segind]
        if pos != segstart:
            # split segment
            ref_chr, ref_s, ref_e, ref_strand = self.segments[segind].ref
            if ref_strand == "+":
                ref_1 = GRange(ref_chr, ref_s, ref_s + pos - segstart, "+")
                ref_2 = GRange(ref_chr, ref_s + pos - segstart, ref_e, "+")
            else:
                ref_1 = GRange(ref_chr, ref_e - (pos - segstart), ref_e, "-")
                ref_2 = GRange(ref_chr, ref_s, ref_e - (pos - segstart), "-")

            self.segments[segind] = LGRange(ref_1.end - ref_1.start, ref_1)
            self.segments.insert(segind + 1, LGRange(ref_2.end - ref_2.start, ref_2))
        self.coord_sync()

    def __add__(self, b):
        a = deepcopy(self)
        a.segments = a.segments + b.segments
        alen = len(a.coord_points)
        a.coord_points = a.coord_points + b.coord_points[1:]
        for i in range(alen, len(a.coord_points)):
            a.coord_points[i] += a.coord_points[alen - 1]
        return a

    def duplicate(self, start, end):
        # start and end in cgenome coordinates
        self._split(start)
        self._split(end)

        ind_s = bisect(self.coord_points, start) - 1
        ind_e = bisect(self.coord_points, end) - 1

        for i, seg in enumerate(self.segments[ind_s:ind_e]):
            self.segments.insert(ind_e + i, deepcopy(seg))
        self.coord_sync()

    def insert(self, start, length, strand="+", name=None):
        self._split(start)
        ind_s = bisect(self.coord_points, start) - 1
        if not name:
            name = "ins" + str(start) + "_" + str(length)
        self.segments.insert(ind_s, LGRange(length, GRange(name, 0, length, strand)))
        self.coord_sync()

    def delete(self, start, end):
        # start and end in cgenome coordinates
        self._split(start)
        self._split(end)

        ind_s = bisect(self.coord_points, start) - 1
        ind_e = bisect(self.coord_points, end) - 1
        del self.segments[ind_s:ind_e]
        self.coord_sync()

    def invert(self, start, end):
        # start and end in cgenome coordinates
        self._split(start)
        self._split(end)

        ind_s = bisect(self.coord_points, start) - 1
        ind_e = bisect(self.coord_points, end) - 1

        self.segments[ind_s:ind_e] = self.segments[ind_s:ind_e][::-1]
        for i in range(ind_s, ind_e):
            self.segments[i] = LGRange(
                self.segments[i].len,
                GRange(
                    self.segments[i].ref.chr,
                    self.segments[i].ref.start,
                    self.segments[i].ref.end,
                    "-" if self.segments[i].ref.strand == "+" else "-",
                ),
            )
        self.coord_sync()

    def query(self, start, end):
        ind_s = bisect(self.coord_points, start) - 1
        ind_e = bisect(self.coord_points, end)

        ref_coords = [deepcopy(seg.ref) for seg in self.segments[ind_s:ind_e]]

        if ref_coords[0]:
            if ref_coords[0].strand == "+":
                ref_coords[0] = GRange(
                    ref_coords[0].chr,
                    ref_coords[0].start + start - self.coord_points[ind_s],
                    ref_coords[0].end,
                    ref_coords[0].strand,
                )
            else:
                ref_coords[0] = GRange(
                    ref_coords[0].chr,
                    ref_coords[0].start,
                    ref_coords[0].end - (start - self.coord_points[ind_s]),
                    ref_coords[0].strand,
                )

        # when end exceeds length
        if ind_e == len(self.coord_points):
            if end > self.coord_points[-1]:
                print(f"Warning: query end {end} exceed limit {self.coord_points[-1]}!")
        else:
            if ref_coords[-1]:
                if ref_coords[-1].strand == "+":
                    ref_coords[-1] = GRange(
                        ref_coords[-1].chr,
                        ref_coords[-1].start,
                        ref_coords[-1].end - (self.coord_points[ind_e] - end),
                        ref_coords[-1].strand,
                    )
                else:
                    ref_coords[-1] = GRange(
                        ref_coords[-1].chr,
                        ref_coords[-1].start + (self.coord_points[ind_e] - end),
                        ref_coords[-1].end,
                        ref_coords[-1].strand,
                    )

        return ref_coords

    def query_ref(self, chr_name, start, end):
        current_coords = []
        ref_coords = []
        for i, (seglen, refseg) in enumerate(self.segments):
            if refseg.chr == chr_name:
                if start < refseg.end or end >= refseg.start:
                    ref_coords.append(
                        [
                            np.clip(start, refseg.start, refseg.end),
                            np.clip(end, refseg.start, refseg.end),
                        ]
                    )
                    if refseg.strand == "+":
                        current_coords.append(
                            [
                                self.coord_points[i] + np.clip(start - refseg.start, 0, seglen),
                                self.coord_points[i] + np.clip(end - refseg.start, 0, seglen),
                                "+",
                            ]
                        )
                    else:
                        current_coords.append(
                            [
                                self.coord_points[i + 1] - np.clip(start - refseg.start, 0, seglen),
                                self.coord_points[i + 1] - np.clip(end - refseg.start, 0, seglen),
                                "-",
                            ]
                        )

        return ref_coords, current_coords

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.query(key.start, key.stop)


def process_anno(anno_scaled, base=0, window_radius=16000000):
    """
    Process annotations to the format used by Orca plotting
    functions such as `genomeplot` and `genomeplot_256Mb`.

    Parameters
    ----------
    anno_scaled : list(list(...))
        List of annotations. Each annotation can be a region specified by
        `[start: int, end: int, info:str]` or a position specified by 
        `[pos: int, info:str]`. 
        Acceptable info strings for region currently include color names for 
        matplotlib. Acceptable info strings for position are currently 
        'single' or 'double', which direct whether the annotation is drawn 
        by single or double lines.
    base : int
        The starting position of the 32Mb (if window_radius is 16000000) 
        or 256Mb (if window_radius is 128000000) region analyzed.
    window_radius : int
        The size of the region analyzed. It must be either 16000000 (32Mb region)
        or 128000000 (256Mb region).

    Returns
    -------
    annotation : list
        Processed annotations with coordinates transformed to relative coordinate
        in the range of 0-1.
    """
    annotation = []
    for r in anno_scaled:
        if len(r) == 3:
            annotation.append(
                [(r[0] - base) / (window_radius * 2), (r[1] - base) / (window_radius * 2), r[2],]
            )
        elif len(r) == 2:
            annotation.append([(r[0] - base) / (window_radius * 2), r[1]])
        else:
            raise ValueError
    return annotation


def coord_clip(pos, chrlen, binsize=128000, window_radius=16000000):
    """
    Clip the coordinate to make sure that full window
    centered at the coordinate to stay within chromosome boundaries.
    coord_clip also try to preserve the relative position of the coordinate
    to the grid as specified by binsize whenever possible.

    Parameters
    ----------
    x : int or numpy.ndarray
        Coordinates to round.
    gridsize : int
        The gridsize to round by

    Returns
    -------
    int
        The clipped coordinate
    """
    if pos < binsize or pos > chrlen - binsize:
        return np.clip(pos, window_radius, chrlen - window_radius)
    else:
        if (chrlen - window_radius) % binsize > pos % binsize:
            endclip = chrlen - window_radius - ((chrlen - window_radius) % binsize - pos % binsize)
        else:
            endclip = (
                chrlen
                - window_radius
                - binsize
                - ((chrlen - window_radius) % binsize - pos % binsize)
            )

        return np.clip(pos, window_radius + pos % binsize, endclip)


def coord_round(x, gridsize=4000):
    """
    Round coordinate to multiples of gridsize.

    Parameters
    ----------
    x : int or numpy.ndarray
        Coordinates to round.
    gridsize : int
        The gridsize to round by

    Returns
    -------
    int
        The rounded coordinate
    """
    return x - x % gridsize
