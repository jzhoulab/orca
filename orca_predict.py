"""
This module provides functions for using Orca models for various
types of the predictions. This is the main module that you need for 
interacting with Orca models.

To use any of the prediction functions, `load_resources` has to be
called first to load the necessary resources.

The coordinates used in Orca are 0-based, inclusive for the start
coordinate and exclusive for the end coordinate, consistent with
python conventions. 
"""
import os
import pathlib

import numpy as np
import torch
from scipy.stats import spearmanr

from selene_utils2 import MemmapGenome, Genomic2DFeatures
import selene_sdk
from selene_sdk.sequences import Genome

from orca_models import H1esc, Hff, H1esc_1M, Hff_1M, H1esc_256M, Hff_256M
from orca_utils import (
    genomeplot,
    genomeplot_256Mb,
    StructuralChange2,
    process_anno,
    coord_round,
    coord_clip,
)

ORCA_PATH = str(pathlib.Path(__file__).parent.absolute())
model_dict_global, target_dict_global = {}, {}

def load_resources(models=["32M"], use_cuda=True, use_memmapgenome=True):
    """
    Load resources for Orca predictions including the specified 
    Orca models and hg38 reference genome. It also creates Genomic2DFeatures
    objects for experimental micro-C  datasets (for comparison with prediction).
    Load resourced are accessible as global variables.
    
    The list of globl variables generated is here:

    Global Variables
    ----------------
    hg38 : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        If `use_memmapgenome==True` and the resource file for hg38 mmap exists,
        use MemmapGenome instead of Genome.
    h1esc : orca_models.H1esc
        1-32Mb Orca H1-ESC model
    hff : orca_models.Hff
        1-32Mb Orca HFF model
    h1esc_256m : orca_models.H1esc_256M
        32-256Mb Orca H1-ESC model
    hff_256m : orca_models.Hff_256M
        32-256Mb Orca HFF model
    h1esc_1m : orca_models.H1esc_1M
        1Mb Orca H1-ESC model
    hff_1m : orca_models.Hff_1M
        1Mb Orca HFF model
    target_h1esc : selene_utils2.Genomic2DFeatures
        Genomic2DFeatures object that load H1-ESC micro-C dataset 4DNFI9GMP2J8
        at 4kb resolution, used for comparison with 1-32Mb models.
    target_hff : selene_utils2.Genomic2DFeatures
        Genomic2DFeatures object that load HFF micro-C dataset 4DNFI643OYP9
        at 4kb resolution, used for comparison with 1-32Mb models.
    target_h1esc_256m : selene_utils2.Genomic2DFeatures
        Genomic2DFeatures object that load H1-ESC micro-C dataset 4DNFI9GMP2J8
        at 32kb resolution, used for comparison with 32-256Mb models.
    target_hff_256m : selene_utils2.Genomic2DFeatures
        Genomic2DFeatures object that load HFF micro-C dataset 4DNFI643OYP9
        at 32kb resolution, used for comparison with 32-256Mb models.
    target_h1esc_1m : selene_utils2.Genomic2DFeatures
        Genomic2DFeatures object that load H1-ESC micro-C dataset 4DNFI9GMP2J8
        at 32kb resolution, used for comparison with 1Mb models.
    target_hff_1m : selene_utils2.Genomic2DFeatures
        Genomic2DFeatures object that load HFF micro-C dataset 4DNFI643OYP9
        at 1kb resolution, used for comparison with 1Mb models.
    target_available : bool
        Indicate whether the micro-C dataset resource file is available.

    Parameters
    ----------
    models : list(str)
        List of model types to load, supported model types includes
        "32M", "256M", "1M", corresponding to 1-32Mb, 32-256Mb, and 1Mb
        models. Lower cases are also accepted.  
    use_cuda : bool, optional
        Default is True. If true, loaded models are moved to GPU.
    use_memmapgenome : bool, optional
        Default is True. If True and the resource file for hg38 mmap exists,
        use MemmapGenome instead of Genome.

 
"""
    global hg38, target_hff, target_h1esc, target_hff_256m, target_h1esc_256m, target_hff_1m, target_h1esc_1m, target_available

    if "32M" in models or "32m" in models:
        global h1esc, hff
        h1esc = H1esc()
        h1esc.eval()
        hff = Hff()
        hff.eval()
        if use_cuda:
            h1esc.cuda()
            hff.cuda()
        else:
            h1esc.cpu()
            hff.cpu()
        model_dict_global["h1esc"] = h1esc
        model_dict_global["hff"] = hff

    if "1M" in models or "1m" in models:
        global h1esc_1m, hff_1m
        h1esc_1m = H1esc_1M()
        h1esc_1m.eval()
        hff_1m = Hff_1M()
        hff_1m.eval()
        if use_cuda:
            h1esc_1m.cuda()
            hff_1m.cuda()
        else:
            h1esc_1m.cpu()
            hff_1m.cpu()
        model_dict_global["h1esc_1m"] = h1esc_1m
        model_dict_global["hff_1m"] = hff_1m

    if "256M" in models or "256m" in models:
        global h1esc_256m, hff_256m
        h1esc_256m = H1esc_256M()
        h1esc_256m.eval()
        hff_256m = Hff_256M()
        hff_256m.eval()
        if use_cuda:
            h1esc_256m.cuda()
            hff_256m.cuda()
        else:
            h1esc_256m.cpu()
            hff_256m.cpu()
        model_dict_global["h1esc_256m"] = h1esc_256m
        model_dict_global["hff_256m"] = hff_256m

    if (
        use_memmapgenome
        and pathlib.Path("/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap").exists()
    ):
        hg38 = MemmapGenome(
            input_path=ORCA_PATH + "/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
            memmapfile=ORCA_PATH + "/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
        )
    else:
        hg38 = Genome(
            input_path=ORCA_PATH + "/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
        )

    target_available = True
    if os.path.exists(ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool"):
        target_hff = Genomic2DFeatures(
            [ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool::/resolutions/4000"],
            ["r4000"],
            (8000, 8000),
            cg=True,
        )
        target_hff_256m = Genomic2DFeatures(
            [ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool::/resolutions/32000"],
            ["r32000"],
            (8000, 8000),
            cg=True,
        )
        target_hff_1m = Genomic2DFeatures(
            [ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool::/resolutions/1000"],
            ["r1000"],
            (8000, 8000),
            cg=True,
        )
        target_dict_global['hff'] = target_hff
        target_dict_global['hff_256m'] = target_hff_256m
        target_dict_global['hff_1m'] = target_hff_1m
    else:
        target_available = False

    if os.path.exists(ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool"):
        target_h1esc = Genomic2DFeatures(
            [ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/4000"],
            ["r4000"],
            (8000, 8000),
            cg=True,
        )

        target_h1esc_256m = Genomic2DFeatures(
            [ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/32000"],
            ["r32000"],
            (8000, 8000),
            cg=True,
        )

        target_h1esc_1m = Genomic2DFeatures(
            [ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/1000"],
            ["r1000"],
            (8000, 8000),
            cg=True,
        )
        target_dict_global['h1esc'] = target_h1esc
        target_dict_global['h1esc_256m'] = target_h1esc_256m
        target_dict_global['h1esc_1m'] = target_h1esc_1m
    else:
        target_available = False


def genomepredict(
    sequence, mchr, mpos=-1, wpos=-1, models=["h1esc", "hff"], targets=None, annotation=None, use_cuda=True, nan_thresh=1,
):
    """Multiscale prediction for a 32Mb sequence
    input, zooming into the position specified when generating a series
    of 32Mb, 16Mb, 8Mb, 4Mb, 2Mb and 1Mb predictions with increasing
    resolutions (up to 4kb). This function also processes 
    information used only for plotting including targets and annotation.

    For larger sequence and interchromosomal predictions, you can use 
    256Mb input with genomepredict_256Mb.

    Parameters
    ----------
    sequence : numpy.ndarray
        One-hot sequence encoding of shape 1 x 4 x 32000000. 
        The encoding can be generated with `selene_sdk.Genome.sequence_to_encoding()`.
    mchr : str
        Chromosome name. This is used for annotation purpose only.
    mpos : int, optional
        The coordinate to zoom into for multiscale prediction.
    wpos : int, optional
        The coordinate of the center position of the sequence, which is
        start position + 16000000.
    models : list(torch.nn.Module or str), optional
        Models to use. Default is H1-ESC and HFF Orca models.
    targets : list(numpy.ndarray), optional
        The observed balanced contact matrices from the
        32Mb region. Used only for plotting when used with genomeplot. The length and
        order of the list of targets should match the models specified (default is 
        H1-ESC and HFF Orca models).
        The dimensions of the arrays should be 8000 x 8000 (1kb resolution).
    annotation : str or None, optional
        List of annotations for plotting. The annotation can be generated with
        See orca_utils.process_anno and see its documentation for more details.
    use_cuda : bool, optional
        Default is True. If False, use CPU.
    nan_thresh : int, optional
        Default is 1. Specify the threshold of the proportion of NaNs values 
        allowed during downsampling for the observed matrices. Only relevant for plotting. 
        The lower resolution observed matrix value are computed by averaging multiple 
        bins into one. By default, we allow missing values and only average over the 
        non-missing values, and the values with more than the specified proprotion 
        of missing values will be filled with NaN.
        
    Returns
    ----------
    output : dict
        Result dictionary that can be used as input for genomeplot. The dictionary
        has the following keys:
            - predictions : list(list(numpy.ndarray), list(numpy.ndarray))
                Multi-level predictions for H1-ESC and HFF cell types.
            - experiments : list(list(numpy.ndarray), list(numpy.ndarray))
                Observations for H1-ESC and HFF cell types that matches the predictions.
                Exists if `targets` is specified.
            - normmats : list(list(numpy.ndarray), list(numpy.ndarray))
                Background distance-based expected balanced contact matrices for 
                H1-ESC and HFF cell types that matches the predictions.
            - start_coords : list(int)
                Start coordinates for the prediction at each level.
            - end_coords : list(int)
                End coordinates for the prediction at each level.
            - chr : str
                The chromosome name.
            - annos : list(list(...))
                Annotation information. The format is as outputed by orca_utils.process_anno
                Exists if `annotation` is specified.


    """
    model_objs = []
    for m in models:
        if isinstance(m, torch.nn.Module):
            model_objs.append(m)
        else:
            try:
                if m in model_dict_global:
                    model_objs.append(model_dict_global[m])
            except KeyError:
                load_resources(models=["32M"], use_cuda=use_cuda)
                if m in model_dict_global:
                    model_objs.append(model_dict_global[m])
    models = model_objs
    n_models = len(models)

    with torch.no_grad():
        allpreds = []
        allstarts = []
        if targets:
            alltargets = []
        if annotation is not None:
            allannos = []

        for iii, seq in enumerate(
            [
                torch.FloatTensor(sequence),
                torch.FloatTensor(sequence[:, ::-1, ::-1].copy()),
            ]
        ):
            for ii, model in enumerate(models):
                if targets and iii == 0:
                    target = targets[ii]
                (encoding1, encoding2, encoding4, encoding8, encoding16, encoding32,) = model.net(
                    model.net0(torch.Tensor(seq.float()).transpose(1, 2).cuda())
                    if use_cuda
                    else model.net0(torch.Tensor(seq.float()).transpose(1, 2))
                )

                encodings = {
                    1: encoding1,
                    2: encoding2,
                    4: encoding4,
                    8: encoding8,
                    16: encoding16,
                    32: encoding32,
                }

                def eval_step(level, start, coarse_pred=None):
                    distenc = torch.log(
                        torch.FloatTensor(model.normmats[level][(None,)*(4-model.normmats[level].ndim)]).cuda()
                        if use_cuda
                        else torch.FloatTensor(model.normmats[level][(None,)*(4-model.normmats[level].ndim)])
                    ).expand(sequence.shape[0], -1, -1, -1)
                    if coarse_pred is not None:
                        if level == 1:
                            pred = model.denets[level].forward(
                                encodings[level][
                                    :, :, int(start / level) : int(start / level) + 250
                                ],
                                distenc,
                                coarse_pred,
                            ) + model.denet_1_pt.forward(
                                encodings[level][
                                    :, :, int(start / level) : int(start / level) + 250
                                ]
                            )
                        else:
                            pred = model.denets[level].forward(
                                encodings[level][
                                    :, :, int(start / level) : int(start / level) + 250
                                ],
                                distenc,
                                coarse_pred,
                            )
                    else:
                        pred = model.denets[level].forward(
                            encodings[level][:, :, int(start / level) : int(start / level) + 250],
                            distenc,
                        )

                    return pred

                preds = []
                starts = [0]
                if targets and iii == 0:
                    ts = []
                if annotation is not None and iii == 0:
                    annos = []
                for j, level in enumerate([32, 16, 8, 4, 2, 1]):
                    if j == 0:
                        pred = eval_step(level, starts[j])
                    else:
                        pred = eval_step(
                            level,
                            starts[j],
                            preds[j - 1][
                                :,
                                :,
                                start_index : start_index + 125,
                                start_index : start_index + 125,
                            ],
                        )

                    if targets and iii == 0:
                        target_r = np.nanmean(
                            np.nanmean(
                                np.reshape(
                                    target[
                                        :,
                                        starts[j] : starts[j] + 250 * level,
                                        starts[j] : starts[j] + 250 * level,
                                    ].numpy(),
                                    (target.shape[0], 250, level, 250, level),
                                ),
                                axis=4,
                            ),
                            axis=2,
                        )
                        target_nan = np.mean(
                            np.mean(
                                np.isnan(
                                    np.reshape(
                                        target[
                                            :,
                                            starts[j] : starts[j] + 250 * level,
                                            starts[j] : starts[j] + 250 * level,
                                        ].numpy(),
                                        (target.shape[0], 250, level, 250, level),
                                    )
                                ),
                                axis=4,
                            ),
                            axis=2,
                        )
                        target_r[target_nan > nan_thresh] = np.nan
                        target_np = np.log(
                            (target_r + model.epss[level])
                            / (model.normmats[level] + model.epss[level])
                        )[0, 0:, 0:]
                        ts.append(target_np)

                    if annotation is not None and iii == 0:
                        newstart = starts[j] / 8000.0
                        newend = (starts[j] + 250 * level) / 8000.0
                        anno_r = []
                        for r in annotation:
                            if len(r) == 3:
                                if not (r[0] >= newend or r[1] <= newstart):
                                    anno_r.append(
                                        (
                                            np.fmax((r[0] - newstart) / (newend - newstart), 0,),
                                            np.fmin((r[1] - newstart) / (newend - newstart), 1,),
                                            r[2],
                                        )
                                    )
                            else:
                                if r[0] >= newstart and r[0] < newend:
                                    anno_r.append(((r[0] - newstart) / (newend - newstart), r[1]))
                        annos.append(anno_r)

                    if iii == 0:
                        start_index = int(
                            np.clip(
                                np.floor(
                                    (
                                        (mpos - level * 1000000 / 4)
                                        - (wpos - 16000000 + starts[j] * 4000)
                                    )
                                    / (4000 * level)
                                ),
                                0,
                                125,
                            )
                        )
                    else:
                        start_index = int(
                            np.clip(
                                np.ceil(
                                    (
                                        (wpos + 16000000 - starts[j] * 4000)
                                        - (mpos + level * 1000000 / 4)
                                    )
                                    / (4000 * level)
                                ),
                                0,
                                125,
                            )
                        )

                    starts.append(starts[j] + start_index * level)
                    preds.append(pred)

                allpreds.append(preds)
                if iii == 0:
                    if targets:
                        alltargets.append(ts)
                    if annotation is not None:
                        allannos.append(annos)
                    allstarts.append(starts[:-1])

    output = {}
    output["predictions"] = [[] for _ in range(n_models)]
    for i in range(n_models):
        for j in range(len(allpreds[i])):
            if allpreds[i][j].shape[1] == 1:
                output["predictions"][i].append(
                    allpreds[i][j].cpu().detach().numpy()[0, 0, :, :] * 0.5
                    + allpreds[i + n_models][j].cpu().detach().numpy()[0, 0, ::-1, ::-1] * 0.5
                )
            else:
                output["predictions"][i].append(
                    allpreds[i][j].cpu().detach().numpy()[0, :, :, :] * 0.5
                    + allpreds[i + n_models][j].cpu().detach().numpy()[0, :, ::-1, ::-1] * 0.5
                )
    if targets:
        output["experiments"] = alltargets
    else:
        output["experiments"] = None
    output["start_coords"] = [wpos - 16000000 + s * 4000 for s in allstarts[0]]
    output["end_coords"] = [
        int(output["start_coords"][ii] + 32000000 / 2 ** (ii)) for ii in range(6)
    ]
    output["chr"] = mchr
    if annotation is not None:
        output["annos"] = allannos[0]
    else:
        output["annos"] = None
    output["normmats"] = [
        [model.normmats[ii] for ii in [32, 16, 8, 4, 2, 1]] for model in models
    ]
    return output


def genomepredict_256Mb(
    sequence,
    mchr,
    normmats,
    chrlen,
    mpos=-1,
    wpos=-1,
    models=["h1esc_256m", "hff_256m"],
    targets=None,
    annotation=None,
    padding_chr=None,
    use_cuda=True,
    nan_thresh=1,
):
    """Multiscale prediction for a 256Mb sequence
    input, zooming into the position specified when generating a series
    of 256Mb, 128Mb, 64Mb, and 32Mb predictions with increasing
    resolutions (up to 128kb). This function also processes 
    information used only for plotting including targets and annotation.

    This function accepts multichromosal input sequence. Thus it needs an
    extra input `normmats` to encode the chromosomal information. See documentation
    for normmats argument for details.

    Parameters
    ----------
    sequence : numpy.ndarray
        One-hot sequence encoding of shape 1 x 4 x 256000000. 
        The encoding can be generated with `selene_sdk.Genome.sequence_to_encoding()`.
    mchr : str
        The chromosome name of the first chromosome included in the seqeunce. 
        This is used for annotation purpose only.
    normmats : list(numpy.ndarray)
        A list of distance-based background matrices for H1-ESC and HFF.The
        normmats contains arrays with dimensions 8000 x 8000  (32kb resolution). 
        Interchromosomal interactions are filled with the expected balanced contact
        score for interchromomsal interactions.
    chrlen : int
        The coordinate of the end of the first chromosome in the input, which is the 
        chromosome that will be zoomed into.
    mpos : int, optional
        Default is -1. The coordinate to zoom into for multiscale prediction. If neither
        `mpos` nor `wpos` are specified, it zooms into the center of the input by default.
    wpos : int, optional
        Default is -1. The coordinate of the center position of the sequence, which is
        start position + 16000000. If neither `mpos` nor `wpos` are specified, it zooms 
        into the center of the input by default.
    models : list(torch.nn.Module or str), optional
        Models to use. Default is H1-ESC(256Mb) and HFF(256Mb) Orca models.
    targets : list(numpy.ndarray), optional
        The observed balanced contact matrices from the 256Mb sequence. 
        Used only for plotting when used with genomeplot. The length and
        order of the list of targets should match the models specified (default is 
        H1-ESC and HFF Orca models). The dimensions of the arrays should be 
        8000 x 8000 (32kb resolution).
    annotation : str or None, optional
        Default is None. List of annotations for plotting. The annotation can be generated with
        See orca_utils.process_anno and see its documentation for more details.
    padding_chr : str, None, optional
        Default is None. Name of the padding chromosome after the first. Used for annotation
        only. TODO: be more flexible in the support for multiple chromosomes.
    use_cuda : bool, optional
        Default is True. If False, use CPU.
    nan_thresh : int, optional
        Default is 1. Specify the threshold of the proportion of NaNs values 
        allowed during downsampling for the observed matrices. Only relevant for plotting. 
        The lower resolution observed matrix value are computed by averaging multiple 
        bins into one. By default, we allow missing values and only average over the 
        non-missing values, and the values with more than the specified proprotion 
        of missing values will be filled with NaN.
        
    Returns
    ----------
    output : dict
        Result dictionary that can be used as input for genomeplot. The dictionary
        has the following keys:
            - predictions : list(list(numpy.ndarray), list(numpy.ndarray))
                Multi-level predictions for H1-ESC and HFF cell types.
            - experiments : list(list(numpy.ndarray), list(numpy.ndarray))
                Observations for H1-ESC and HFF cell types that matches the predictions.
                Exists if `targets` is specified.
            - normmats : list(list(numpy.ndarray), list(numpy.ndarray))
                Background distance-based expected balanced contact matrices for 
                H1-ESC and HFF cell types that matches the predictions.
            - start_coords : list(int)
                Start coordinates for the prediction at each level.
            - end_coords : list(int)
                End coordinates for the prediction at each level.
            - chr : str
                The chromosome name.
            - annos : list(list(...))
                Annotation information. The format is as outputed by orca_utils.process_anno
                Exists if `annotation` is specified.
    """
    model_objs = []
    for m in models:
        if isinstance(m, torch.nn.Module):
            model_objs.append(m)
        else:
            try:
                if m in model_dict_global:
                    model_objs.append(model_dict_global[m])
            except KeyError:
                load_resources(models=["256M"], use_cuda=use_cuda)
                if m in model_dict_global:
                    model_objs.append(model_dict_global[m])
    models = model_objs
    n_models = len(models)
    
    with torch.no_grad():
        allpreds = []
        allstarts = []
        allnormmats = []
        if targets:
            alltargets = []
        if annotation is not None:
            allannos = []

        for iii, seq in enumerate(
            [
                torch.FloatTensor(sequence),
                torch.FloatTensor(sequence[:, ::-1, ::-1].copy()),
            ]
        ):
            for ii, model in enumerate(models):
                normmat = normmats[ii]
                normmat_nan = np.isnan(normmat)
                if np.any(normmat_nan):
                    normmat[normmat_nan] = np.nanmin(normmat[~normmat_nan])
                if targets and iii == 0:
                    target = targets[ii]

                (encoding32, encoding64, encoding128, encoding256) = model.net(
                    model.net1(
                        model.net0(
                            torch.Tensor(seq.float()).transpose(1, 2).cuda()
                            if use_cuda
                            else torch.Tensor(seq.float()).transpose(1, 2)
                        )
                    )[-1]
                )

                encodings = {
                    32: encoding32,
                    64: encoding64,
                    128: encoding128,
                    256: encoding256,
                }

                def eval_step(level, start, coarse_pred=None):
                    distenc = torch.log(
                        torch.FloatTensor(ns[level][None, :, :]).cuda()
                        if use_cuda
                        else torch.FloatTensor(ns[level][None, :, :])
                    ).expand(sequence.shape[0], -1, -1, -1)
                    if coarse_pred is not None:
                        pred = model.denets[level].forward(
                            encodings[level][
                                :, :, int(start / (level // 8)) : int(start / (level // 8)) + 250,
                            ],
                            distenc if iii == 0 else torch.flip(distenc, [2, 3]),
                            coarse_pred,
                        )
                    else:
                        pred = model.denets[level].forward(
                            encodings[level][
                                :, :, int(start / (level // 8)) : int(start / (level // 8)) + 250,
                            ],
                            distenc if iii == 0 else torch.flip(distenc, [2, 3]),
                        )

                    return pred

                preds = []
                starts = [0]
                ns = {}
                if targets and iii == 0:
                    ts = []
                if annotation is not None and iii == 0:
                    annos = []
                for j, level in enumerate([256, 128, 64, 32]):
                    normmat_r = np.nanmean(
                        np.nanmean(
                            np.reshape(
                                normmat[
                                    starts[j] : starts[j] + 250 * level // 8,
                                    starts[j] : starts[j] + 250 * level // 8,
                                ],
                                (1, 250, level // 8, 250, level // 8),
                            ),
                            axis=4,
                        ),
                        axis=2,
                    )
                    ns[level] = normmat_r

                    if j == 0:
                        pred = eval_step(level, starts[j])
                    else:
                        pred = eval_step(
                            level,
                            starts[j],
                            preds[j - 1][
                                :,
                                :,
                                start_index : start_index + 125,
                                start_index : start_index + 125,
                            ],
                        )

                    if targets and iii == 0:
                        target_r = np.nanmean(
                            np.nanmean(
                                np.reshape(
                                    target[
                                        :,
                                        starts[j] : starts[j] + 250 * level // 8,
                                        starts[j] : starts[j] + 250 * level // 8,
                                    ].numpy(),
                                    (target.shape[0], 250, level // 8, 250, level // 8),
                                ),
                                axis=4,
                            ),
                            axis=2,
                        )
                        target_nan = np.mean(
                            np.mean(
                                np.isnan(
                                    np.reshape(
                                        target[
                                            :,
                                            starts[j] : starts[j] + 250 * level // 8,
                                            starts[j] : starts[j] + 250 * level // 8,
                                        ].numpy(),
                                        (target.shape[0], 250, level // 8, 250, level // 8,),
                                    )
                                ),
                                axis=4,
                            ),
                            axis=2,
                        )
                        target_r[target_nan > nan_thresh] = np.nan
                        eps = np.nanmin(normmat_r)
                        target_np = np.log((target_r + eps) / (normmat_r + eps))[0, 0:, 0:]
                        ts.append(target_np)

                    if annotation is not None and iii == 0:
                        newstart = starts[j] / 8000.0
                        newend = (starts[j] + 250 * level // 8) / 8000.0
                        anno_r = []
                        for r in annotation:
                            if len(r) == 3:
                                if not (r[0] >= newend or r[1] <= newstart):
                                    anno_r.append(
                                        (
                                            np.fmax((r[0] - newstart) / (newend - newstart), 0,),
                                            np.fmin((r[1] - newstart) / (newend - newstart), 1,),
                                            r[2],
                                        )
                                    )
                            else:
                                if r[0] >= newstart and r[0] < newend:
                                    anno_r.append(((r[0] - newstart) / (newend - newstart), r[1]))
                        annos.append(anno_r)

                    if iii == 0:
                        proposed_start = (mpos - level * 1000000 / 4) - (
                            wpos - 128000000 + starts[j] * 4000 * 8
                        )
                    else:
                        proposed_start = (mpos - level * 1000000 / 4) - (
                            wpos + 128000000 - starts[j] * 4000 * 8 - level * 1000000
                        )
                    if chrlen is not None:
                        bounds = [
                            0 - (wpos - 128000000),
                            chrlen - level * 1000000 / 2 - (wpos - 128000000),
                        ]
                        if bounds[0] < bounds[1]:
                            proposed_start = np.clip(proposed_start, bounds[0], bounds[1])
                        else:
                            proposed_start = bounds[0]

                    start_index = int(np.clip(np.floor(proposed_start / (4000 * level)), 0, 125,))
                    if iii != 0:
                        start_index = 250 - (start_index + 125)

                    starts.append(starts[j] + start_index * level // 8)
                    preds.append(pred)

                allpreds.append(preds)
                allnormmats.append(ns)
                if iii == 0:
                    if targets:
                        alltargets.append(ts)
                    if annotation is not None:
                        allannos.append(annos)
                    allstarts.append(starts[:-1])

    output = {}

    output["predictions"] = [[] for _ in range(n_models)]
    for i in range(n_models):
        for j in range(len(allpreds[i])):
            if allpreds[i][j].shape[1] == 1:
                output["predictions"][i].append(
                    allpreds[i][j].cpu().detach().numpy()[0, 0, :, :] * 0.5
                    + allpreds[i + n_models][j].cpu().detach().numpy()[0, 0, ::-1, ::-1] * 0.5
                )
            else:
                output["predictions"][i].append(
                    allpreds[i][j].cpu().detach().numpy()[0, :, :, :] * 0.5
                    + allpreds[i + n_models][j].cpu().detach().numpy()[0, :, ::-1, ::-1] * 0.5
                )
    if targets:
        output["experiments"] = alltargets
    else:
        output["experiments"] = None
    output["start_coords"] = [wpos - 128000000 + s * 32000 for s in allstarts[0]]
    output["end_coords"] = [
        np.fmin(int(output["start_coords"][ii] + 256000000 / 2 ** (ii)), chrlen) for ii in range(4)
    ]

    if annotation is not None:
        output["annos"] = allannos[0]
    else:
        output["annos"] = None
    output["chr"] = mchr
    output["padding_chr"] = padding_chr
    output["normmats"] = allnormmats
    return output


def _retrieve_multi(regionlist, genome, target=True, normmat=True, normmat_regionlist=None):
    sequences = []
    for region in regionlist:
        if len(region) == 4:
            chrom, start, end, strand = region
            sequences.append(genome.get_encoding_from_coords(chrom, start, end, strand))
        else:
            chrom, start, end = region
            sequences.append(genome.get_encoding_from_coords(chrom, start, end, "+"))

    sequence = np.vstack(sequences)[None, :, :]

    
    if isinstance(target, list):
        target_objs = target
        has_target = True
    elif target and target_available:
        target_objs = [target_h1esc_256m, target_hff_256m]
        has_target = True
    else:
        has_target = False

    if has_target:
        targets = []
        for target_obj in target_objs:
            targets_ = []
            for region in regionlist:
                if len(region) == 4:
                    chrom, start, end, strand = region
                else:
                    chrom, start, end = region
                    strand = "+"
                t = []
                for region2 in regionlist:
                    if len(region2) == 4:
                        chrom2, start2, end2, strand2 = region2
                    else:
                        chrom2, start2, end2 = region2
                        strand = "+"
                    t.append(
                        target_obj.get_feature_data(
                            chrom, start, end, chrom2=chrom2, start2=start2, end2=end2
                        )
                    )
                    if strand == "-":
                        t[-1] = t[-1][::-1, :]
                    if strand2 == "-":
                        t[-1] = t[-1][:, ::-1]
                targets_.append(t)
            targets_= np.vstack([np.hstack(l) for l in targets_])
            targets.append(targets_)
        targets = [
            torch.FloatTensor(l[None, :, :]) for l in targets
        ]

    if normmat:
        if isinstance(normmat, list):
            normmat_objs = normmat
        else:
            normmat_objs = [h1esc_256m, hff_256m]
        
        if normmat_regionlist is None:
            normmat_regionlist = regionlist

        normmats = []
        for normmat_obj in normmat_objs:
            normmats_ = []
            for chrom, start, end, strand in normmat_regionlist:
                b = []
                for chrom2, start2, end2, strand2 in normmat_regionlist:
                    if chrom2 != chrom:
                        b.append(
                            np.full(
                                (int((end - start) / 32000), int((end2 - start2) / 32000)),
                                normmat_obj.background_trans,
                            )
                        )
                    else:
                        binsize = 32000
                        acoor = np.linspace(start, end, int((end - start) / 32000) + 1)[:-1]
                        bcoor = np.linspace(start2, end2, int((end2 - start2) / 32000) + 1)[:-1]
                        b.append(
                            normmat_obj.background_cis[
                                (np.abs(acoor[:, None] - bcoor[None, :]) / binsize).astype(int)
                            ]
                        )
                        if strand == "-":
                            b[-1] = b[-1][::-1, :]
                        if strand2 == "-":
                            b[-1] = b[-1][:, ::-1]
                normmats_.append(b)
            normmats_ = np.vstack([np.hstack(l) for l in normmats_])
            normmats.append(normmats_)

    datatuple = (sequence,)
    if normmat:
        datatuple = datatuple + (normmats,)
    if has_target:
        datatuple = datatuple + (targets,)
    return datatuple


def process_region(
    mchr,
    mstart,
    mend,
    genome,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=16000000,
    padding_chr="chr1",
    use_cuda=True,
):
    """
    Generate multiscale genome interaction predictions for 
    the specified region. 

    Parameters
    ----------
    mchr : str
        The chromosome name of the first segment
    mstart : int
        The start coordinate of the region.
    mend : ind
        The end coordinate of the region.
    genome : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        The reference genome object to extract sequence from
    custom_models : list(torch.nn.Module or str) or None, optional
        Models to use instead of the default H1-ESC and HFF Orca models.
        Default is None.
    target : list(selene_utils2.Genomic2DFeatures or str) or bool, optional
        If specified as list, use this list of targets to retrieve experimental
        data (for plotting only). Default is True and will use micro-C data 
        for H1-ESC and HFF cells (4DNFI9GMP2J8, 4DNFI643OYP9) that correspond
        to the default models.
    file : str or None, optional
        Default is None. The output file prefix.
    show_genes : bool, optional
        Default is True. If True, generate gene annotation visualization
        file in pdf format that matches the windows of multiscale predictions.
    show_tracks : bool, optional
        Default is False. If True, generate chromatin tracks visualization
        file in pdf format that matches the windows of multiscale predictions.
    window_radius : int, optional
        Default is 16000000. The acceptable values are 16000000 which selects
        the 1-32Mb models or 128000000 which selects the 32-256Mb models.
    padding_chr : str, optional
        Default is "chr1". If window_radius is 128000000, padding is generally 
        needed to fill the sequence to 256Mb. The padding sequence will be 
        extracted from the padding_chr.
    use_cuda : bool, optional
        Default is True. Use CPU if False.

    Returns
    -------
    outputs_ref_l, outputs_ref_r, outputs_alt : dict, dict, dict
        Reference allele predictions zooming into the left boundary of the
        duplication,
        Reference allele predictions zooming into the right boundary of the
        duplication,
        Alternative allele predictions zooming into the duplication breakpoint.
        The returned results are in the format of dictonaries 
        containing the prediction outputs and other 
        retrieved information. These dictionaries can be directly used as
        input to genomeplot or genomeplot_256Mb. See documentation of `genomepredict` or `genomepredict_256Mb` for
        details of the dictionary content.

    """
    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()
    mpos = int((int(mstart) + int(mend)) / 2)

    if custom_models is None:
        if window_radius == 16000000:
            models = ["h1esc", "hff"]
        elif window_radius == 128000000:
            models = ["h1esc_256m", "hff_256m"]
        else:
            raise ValueError(
                "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
            )
    else:
        models = custom_models

    if target:
        try:
            if target == True:
                if window_radius == 16000000:
                    target = ["h1esc", "hff"]
                elif window_radius == 128000000:
                    target = ["h1esc_256m", "hff_256m"]
            target = [t if isinstance(t, Genomic2DFeatures) else target_dict_global[t] for t in target]
        except KeyError:
            target = False

    if window_radius == 16000000:
        wpos = coord_clip(mpos, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        mchr, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                )
                for t in target
            ]
        else:
            targets = None
    elif window_radius == 128000000:
        chrlen_round = chrlen - chrlen % 32000
        wpos = 128000000
        if target:
            sequence, normmats, targets = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
        else:
            sequence, normmats = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
            targets = None
    else:
        raise ValueError(
            "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
        )

    if mstart - mend < 2 * window_radius:
        anno_scaled = process_anno(
            [
                [
                    np.clip(mstart, wpos - window_radius, wpos + window_radius),
                    np.clip(mend, wpos - window_radius, wpos + window_radius),
                    "black",
                ]
            ],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    else:
        anno_scaled = None

    if window_radius == 128000000:
        outputs_ref = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mpos,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
    else:
        outputs_ref = genomepredict(
            sequence, mchr, mpos, wpos, annotation=anno_scaled, models=models, targets=targets, use_cuda=use_cuda,
        )
    if file is not None:
        if window_radius == 128000000:
            genomeplot_256Mb(
                outputs_ref, show_coordinates=True, file=file + ".256m.pdf",
            )
        else:
            genomeplot(
                outputs_ref,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".pdf",
            )
    return outputs_ref


def process_dup(
    mchr,
    mstart,
    mend,
    genome,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=16000000,
    padding_chr="chr1",
    use_cuda=True,
):
    """
    Generate multiscale genome interaction predictions for 
    an duplication variant. 

    Parameters
    ----------
    mchr : str
        The chromosome name of the first segment
    mstart : int
        The start coordinate of the duplication.
    mend : ind
        The end coordinate of the duplication.
    genome : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        The reference genome object to extract sequence from
    custom_models : list(torch.nn.Module or str) or None, optional
        Models to use instead of the default H1-ESC and HFF Orca models.
        Default is None.
    target : list(selene_utils2.Genomic2DFeatures or str) or bool, optional
        If specified as list, use this list of targets to retrieve experimental
        data (for plotting only). Default is True and will use micro-C data 
        for H1-ESC and HFF cells (4DNFI9GMP2J8, 4DNFI643OYP9) that correspond
        to the default models.
    file : str or None, optional
        Default is None. The output file prefix.
    show_genes : bool, optional
        Default is True. If True, generate gene annotation visualization
        file in pdf format that matches the windows of multiscale predictions.
    show_tracks : bool, optional
        Default is False. If True, generate chromatin tracks visualization
        file in pdf format that matches the windows of multiscale predictions.
    window_radius : int, optional
        Default is 16000000. The acceptable values are 16000000 which selects
        the 1-32Mb models or 128000000 which selects the 32-256Mb models.
    padding_chr : str, optional
        Default is "chr1". If window_radius is 128000000, padding is generally 
        needed to fill the sequence to 256Mb. The padding sequence will be 
        extracted from the padding_chr.
    use_cuda : bool, optional
        Default is True. Use CPU if False.

    Returns
    -------
    outputs_ref_l, outputs_ref_r, outputs_alt : dict, dict, dict
        Reference allele predictions zooming into the left boundary of the
        duplication,
        Reference allele predictions zooming into the right boundary of the
        duplication,
        Alternative allele predictions zooming into the duplication breakpoint.
        The returned results are in the format of dictonaries 
        containing the prediction outputs and other 
        retrieved information. These dictionaries can be directly used as
        input to genomeplot or genomeplot_256Mb. See documentation of `genomepredict` or `genomepredict_256Mb` for
        details of the dictionary content.
    """
    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    if custom_models is None:
        if window_radius == 16000000:
            models = ["h1esc", "hff"]
        elif window_radius == 128000000:
            models = ["h1esc_256m", "hff_256m"]
        else:
            raise ValueError(
                "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
            )
    else:
        models = custom_models

    if target:
        try:
            if target == True:
                if window_radius == 16000000:
                    target = ["h1esc", "hff"]
                elif window_radius == 128000000:
                    target = ["h1esc_256m", "hff_256m"]
            target = [t if isinstance(t, Genomic2DFeatures) else target_dict_global[t] for t in target]
        except KeyError:
            target = False

    # ref.l
    if window_radius == 16000000:
        wpos = coord_clip(mstart, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        mchr, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None
    elif window_radius == 128000000:
        chrlen_round = chrlen - chrlen % 32000
        wpos = 128000000
        if target:
            sequence, normmats, targets = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
        else:
            sequence, normmats = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
            targets = None
    else:
        raise ValueError(
            "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
        )

    if wpos + window_radius > mend:
        anno_scaled = process_anno(
            [[mstart, mend, "black"]], base=wpos - window_radius, window_radius=window_radius
        )
    else:
        anno_scaled = process_anno(
            [[mstart, wpos + window_radius, "black"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 128000000:
        outputs_ref_l = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mstart,
            wpos,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            models=models,
            targets=targets,
            use_cuda=use_cuda,
        )
    else:
        outputs_ref_l = genomepredict(
            sequence,
            mchr,
            mstart,
            wpos,
            annotation=anno_scaled,
            models=models,
            targets=targets,
            use_cuda=use_cuda,
        )
    if file is not None:
        if window_radius == 128000000:
            genomeplot_256Mb(
                outputs_ref_l, show_coordinates=True, file=file + ".ref.l.256m.pdf",
            )
        else:
            genomeplot(
                outputs_ref_l,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref.l.pdf",
            )

    # ref.r
    if window_radius == 16000000:
        wpos = coord_clip(mend, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        mchr, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None

    if wpos - window_radius < mstart:
        anno_scaled = process_anno(
            [[mstart, mend, "black"]], base=wpos - window_radius, window_radius=window_radius
        )
    else:
        anno_scaled = process_anno(
            [[wpos - window_radius, mend, "black"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 16000000:
        outputs_ref_r = genomepredict(
            sequence, mchr, mend, wpos, models=models, annotation=anno_scaled, targets=targets, use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot(
                outputs_ref_r,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref.r.pdf",
            )
    else:
        outputs_ref_r = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mend,
            wpos,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            models=models,
            targets=targets,
            use_cuda=use_cuda,
        )
        genomeplot_256Mb(
            outputs_ref_r, show_coordinates=True, file=file + ".ref.r.256m.pdf",
        )

    # alt (r)
    s = StructuralChange2(mchr, chrlen)
    s.duplicate(mstart, mend)
    chrlen_alt = chrlen + mend - mstart
    if window_radius == 16000000:
        wpos = coord_clip(mend, chrlen_alt)
        sequence = []
        for chrm, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
            seq = genome.get_encoding_from_coords(chrm, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1]
            else:
                seq = seq[None, :, :]
            sequence.append(seq)
        sequence = np.concatenate(sequence, axis=1)
    else:
        chrlen_alt_round = chrlen_alt - chrlen_alt % 32000
        if chrlen_alt_round < 256000000:
            wpos = 128000000
            (sequence, normmats) = _retrieve_multi(
                list(s[0:chrlen_alt_round]) + [[padding_chr, 0, 256000000 - chrlen_alt_round, "+"]],
                genome,
                target=False,
                normmat=True,
                normmat_regionlist=[
                    [mchr, 0, chrlen_alt_round, "+"],
                    [padding_chr, 0, 256000000 - chrlen_alt_round, "+"],
                ],
            )
        else:
            wpos = coord_clip(mend, chrlen_alt_round, window_radius=128000000)
            (sequence, normmats) = _retrieve_multi(
                list(s[wpos - window_radius : wpos + window_radius]),
                genome,
                target=False,
                normmat=True,
                normmat_regionlist=[[mchr, wpos - window_radius, wpos + window_radius, "+"]],
            )

    if wpos - window_radius < mstart and mend + mend - mstart < wpos + window_radius:
        anno_scaled = process_anno(
            [[mstart, mend, "black"], [mend, mend + mend - mstart, "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    elif wpos - window_radius >= mstart and mend + mend - mstart < wpos + window_radius:
        anno_scaled = process_anno(
            [[wpos - window_radius, mend, "black"], [mend, mend + mend - mstart, "gray"],],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    elif wpos - window_radius < mstart and mend + mend - mstart >= wpos + window_radius:
        anno_scaled = process_anno(
            [[mstart, mend, "black"], [mend, wpos + window_radius, "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    else:
        anno_scaled = process_anno(
            [[wpos - window_radius, mend, "black"], [mend, wpos + window_radius, "gray"],],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 16000000:
        outputs_alt = genomepredict(
            sequence, mchr, mend, wpos, models=models, annotation=anno_scaled, use_cuda=use_cuda
        )
        if file is not None:
            genomeplot(outputs_alt, show_coordinates=True, file=file + ".alt.pdf")
    else:
        outputs_alt = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_alt_round,
            mend,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_alt, show_coordinates=True, file=file + ".alt.256m.pdf",
            )

    return outputs_ref_l, outputs_ref_r, outputs_alt


def process_del(
    mchr,
    mstart,
    mend,
    genome,
    cmap=None,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=16000000,
    padding_chr="chr1",
    use_cuda=True,
):
    """
    Generate multiscale genome interaction predictions for 
    an deletion variant. 

    Parameters
    ----------
    mchr : str
        The chromosome name of the first segment
    mstart : int
        The start coordinate of the deletion.
    mend : ind
        The end coordinate of the deletion.
    genome : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        The reference genome object to extract sequence from
    custom_models : list(torch.nn.Module or str) or None, optional
        Models to use instead of the default H1-ESC and HFF Orca models.
        Default is None.
    target : list(selene_utils2.Genomic2DFeatures or str) or bool, optional
        If specified as list, use this list of targets to retrieve experimental
        data (for plotting only). Default is True and will use micro-C data 
        for H1-ESC and HFF cells (4DNFI9GMP2J8, 4DNFI643OYP9) that correspond
        to the default models.
    file : str or None, optional
        Default is None. The output file prefix.
    show_genes : bool, optional
        Default is True. If True, generate gene annotation visualization
        file in pdf format that matches the windows of multiscale predictions.
    show_tracks : bool, optional
        Default is False. If True, generate chromatin tracks visualization
        file in pdf format that matches the windows of multiscale predictions.
    window_radius : int, optional
        Default is 16000000. The acceptable values are 16000000 which selects
        the 1-32Mb models or 128000000 which selects the 32-256Mb models.
    padding_chr : str, optional
        Default is "chr1". If window_radius is 128000000, padding is generally 
        needed to fill the sequence to 256Mb. The padding sequence will be 
        extracted from the padding_chr.
    use_cuda : bool, optional
        Default is True. Use CPU if False.

    Returns
    -------
    outputs_ref_l, outputs_ref_r, outputs_alt : dict, dict, dict
        Reference allele predictions zooming into the left boundary of the
        deletion,
        Reference allele predictions zooming into the right boundary of the
        deletion,
        Alternative allele predictions zooming into the deletion breakpoint.
        The returned results are in the format of dictonaries 
        containing the prediction outputs and other 
        retrieved information. These dictionaries can be directly used as
        input to genomeplot or genomeplot_256Mb. See documentation of `genomepredict` or `genomepredict_256Mb` for
        details of the dictionary content.
    """
    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    if custom_models is None:
        if window_radius == 16000000:
            models = ["h1esc", "hff"]
        elif window_radius == 128000000:
            models = ["h1esc_256m", "hff_256m"]
        else:
            raise ValueError(
                "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
            )
    else:
        models = custom_models

    if target:
        try:
            if target == True:
                if window_radius == 16000000:
                    target = ["h1esc", "hff"]
                elif window_radius == 128000000:
                    target = ["h1esc_256m", "hff_256m"]
            target = [t if isinstance(t, Genomic2DFeatures) else target_dict_global[t] for t in target]
        except KeyError:
            target = False

    # ref.l
    if window_radius == 16000000:
        wpos = coord_clip(mstart, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        mchr, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None
    elif window_radius == 128000000:
        chrlen_round = chrlen - chrlen % 32000
        wpos = 128000000
        if target:
            sequence, normmats, targets = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
        else:
            sequence, normmats = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
    else:
        raise ValueError(
            "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
        )

    if wpos + window_radius > mend:
        anno_scaled = process_anno(
            [[mstart, mend, "black"]], base=wpos - window_radius, window_radius=window_radius
        )
    else:
        anno_scaled = process_anno(
            [[mstart, wpos + window_radius, "black"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 128000000:
        outputs_ref_l = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mstart,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
    else:
        outputs_ref_l = genomepredict(
            sequence,
            mchr,
            mstart,
            wpos,
            models=models,
            annotation=anno_scaled,
            targets=targets,
            use_cuda=use_cuda,
        )
    if file is not None:
        if window_radius == 128000000:
            genomeplot_256Mb(
                outputs_ref_l, show_coordinates=True, file=file + ".ref.l.256m.pdf",
            )
        else:
            genomeplot(
                outputs_ref_l,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                cmap=cmap,
                file=file + ".ref.l.pdf",
            )

    # ref.r
    if window_radius == 16000000:
        wpos = coord_clip(mend, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        mchr, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None

    if wpos - window_radius < mstart:
        anno_scaled = process_anno(
            [[mstart, mend, "black"]], base=wpos - window_radius, window_radius=window_radius
        )
    else:
        anno_scaled = process_anno(
            [[wpos - window_radius, mend, "black"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 16000000:
        outputs_ref_r = genomepredict(
            sequence, mchr, mend, wpos, models=models, annotation=anno_scaled, targets=targets, use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot(
                outputs_ref_r,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                cmap=cmap,
                file=file + ".ref.r.pdf",
            )
    else:
        outputs_ref_r = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mend,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_ref_r, show_coordinates=True, file=file + ".ref.r.256m.pdf",
            )

    # alt
    s = StructuralChange2(mchr, chrlen)
    s.delete(mstart, mend)
    chrlen_alt = chrlen - (mend - mstart)
    if window_radius == 16000000:
        wpos = coord_clip(mstart, chrlen_alt)
        sequence = []
        for chrm, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
            seq = genome.get_encoding_from_coords(chrm, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1]
            else:
                seq = seq[None, :, :]
            sequence.append(seq)
        sequence = np.concatenate(sequence, axis=1)
    else:
        chrlen_alt_round = chrlen_alt - chrlen_alt % 32000
        wpos = 128000000
        (sequence, normmats) = _retrieve_multi(
            list(s[0:chrlen_alt_round]) + [[padding_chr, 0, 256000000 - chrlen_alt_round, "+"]],
            genome,
            target=False,
            normmat=True,
            normmat_regionlist=[
                [mchr, 0, chrlen_alt_round, "+"],
                [padding_chr, 0, 256000000 - chrlen_alt_round, "+"],
            ],
        )

    anno_scaled = process_anno(
        [[mstart, "double"]], base=wpos - window_radius, window_radius=window_radius
    )

    if window_radius == 16000000:
        outputs_alt = genomepredict(
            sequence, mchr, mstart, wpos, models=models, annotation=anno_scaled, use_cuda=use_cuda
        )
        if file is not None:
            genomeplot(outputs_alt, show_coordinates=True, cmap=cmap, file=file + ".alt.pdf")
    else:
        outputs_alt = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_alt_round,
            mstart,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_alt, show_coordinates=True, file=file + ".alt.256m.pdf",
            )

    return outputs_ref_l, outputs_ref_r, outputs_alt


def process_inv(
    mchr,
    mstart,
    mend,
    genome,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=16000000,
    padding_chr="chr1",
    use_cuda=True,
):
    """
    Generate multiscale genome interaction predictions for 
    an inversion variant. 

    Parameters
    ----------
    mchr : str
        The chromosome name of the first segment
    mstart : int
        The start coordinate of the inversion.
    mend : ind
        The end coordinate of the inversion.
    genome : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        The reference genome object to extract sequence from
    custom_models : list(torch.nn.Module or str) or None, optional
        Models to use instead of the default H1-ESC and HFF Orca models.
        Default is None.
    target : list(selene_utils2.Genomic2DFeatures or str) or bool, optional
        If specified as list, use this list of targets to retrieve experimental
        data (for plotting only). Default is True and will use micro-C data 
        for H1-ESC and HFF cells (4DNFI9GMP2J8, 4DNFI643OYP9) that correspond
        to the default models.
    file : str or None, optional
        Default is None. The output file prefix.
    show_genes : bool, optional
        Default is True. If True, generate gene annotation visualization
        file in pdf format that matches the windows of multiscale predictions.
    show_tracks : bool, optional
        Default is False. If True, generate chromatin tracks visualization
        file in pdf format that matches the windows of multiscale predictions.
    window_radius : int, optional
        Default is 16000000. The acceptable values are 16000000 which selects
        the 1-32Mb models or 128000000 which selects the 32-256Mb models.
    padding_chr : str, optional
        Default is "chr1". If window_radius is 128000000, padding is generally 
        needed to fill the sequence to 256Mb. The padding sequence will be 
        extracted from the padding_chr.
    use_cuda : bool, optional
        Default is True. Use CPU if False.

    Returns
    -------
    outputs_ref_l, outputs_ref_r, outputs_alt_l, outputs_alt_r : dict, dict, dict, dict
        Reference allele predictions zooming into the left boundary of the
        inversion,
        Reference allele predictions zooming into the right boundary of the
        inversion,
        Alternative allele predictions zooming into the left boundary of 
        the inversion,
        Alternative allele prediction zooming into the right boundary of 
        the inversion.
        The returned results are in the format of dictonaries 
        containing the prediction outputs and other 
        retrieved information. These dictionaries can be directly used as
        input to genomeplot or genomeplot_256Mb. See documentation of `genomepredict` or `genomepredict_256Mb` for
        details of the dictionary content.
    """
    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    if custom_models is None:
        if window_radius == 16000000:
            models = ["h1esc", "hff"]
        elif window_radius == 128000000:
            models = ["h1esc_256m", "hff_256m"]
        else:
            raise ValueError(
                "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
            )
    else:
        models = custom_models

    if target:
        try:
            if target == True:
                if window_radius == 16000000:
                    target = ["h1esc", "hff"]
                elif window_radius == 128000000:
                    target = ["h1esc_256m", "hff_256m"]
            target = [t if isinstance(t, Genomic2DFeatures) else target_dict_global[t] for t in target]
        except KeyError:
            target = False

    if window_radius == 16000000:
        wpos = coord_clip(mstart, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        mchr, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None
    elif window_radius == 128000000:
        chrlen_round = chrlen - chrlen % 32000
        wpos = 128000000
        if target:
            sequence, normmats, targets = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
        else:
            sequence, normmats = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
            targets = None
    else:
        raise ValueError(
            "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
        )

    if wpos + window_radius > mend:
        anno_scaled = process_anno(
            [[mstart, mend, "black"]], base=wpos - window_radius, window_radius=window_radius,
        )
    else:
        anno_scaled = process_anno(
            [[mstart, wpos + window_radius, "black"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    if window_radius == 128000000:
        outputs_ref_l = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mstart,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
    else:
        outputs_ref_l = genomepredict(
            sequence,
            mchr,
            mstart,
            wpos,
            models=models,
            annotation=anno_scaled,
            targets=targets,
            use_cuda=use_cuda,
        )
    if file is not None:
        if window_radius == 128000000:
            genomeplot_256Mb(
                outputs_ref_l, show_coordinates=True, file=file + ".ref.l.256m.pdf",
            )
        else:
            genomeplot(
                outputs_ref_l,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref.l.pdf",
            )

    # ref.r
    if window_radius == 16000000:
        wpos = coord_clip(mend, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        mchr, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None

    if wpos - window_radius < mstart:
        anno_scaled = process_anno(
            [[mstart, mend, "black"]], base=wpos - window_radius, window_radius=window_radius,
        )
    else:
        anno_scaled = process_anno(
            [[wpos - window_radius, mend, "black"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 16000000:
        outputs_ref_r = genomepredict(
            sequence, mchr, mend, wpos, models=models, annotation=anno_scaled, targets=targets, use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot(
                outputs_ref_r,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref.r.pdf",
            )
    else:
        outputs_ref_r = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mend,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_ref_r, show_coordinates=True, file=file + ".ref.r.256m.pdf",
            )

    # alt.l
    s = StructuralChange2(mchr, chrlen)
    s.invert(mstart, mend)
    if window_radius == 16000000:
        wpos = coord_clip(mstart, chrlen)
        sequence = []
        for chrm, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
            seq = genome.get_encoding_from_coords(chrm, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1]
            else:
                seq = seq[None, :, :]
            sequence.append(seq)
        sequence = np.concatenate(sequence, axis=1)
    else:
        wpos = 128000000
        (sequence,) = _retrieve_multi(
            list(s[0:chrlen_round]) + [[padding_chr, 0, 256000000 - chrlen_round, "+"]],
            genome,
            target=False,
            normmat=False,
        )

        # normmats are not changed for inversion

    if mend < wpos + window_radius:
        anno_scaled = process_anno(
            [[mstart, mend, "gray"]], base=wpos - window_radius, window_radius=window_radius,
        )
    else:
        anno_scaled = process_anno(
            [[mstart, wpos + window_radius, "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 16000000:
        outputs_alt_l = genomepredict(
            sequence, mchr, mstart, wpos, models=models, annotation=anno_scaled, use_cuda=use_cuda
        )
        if file is not None:
            genomeplot(outputs_alt_l, show_coordinates=True, file=file + ".alt.l.pdf")
    else:
        outputs_alt_l = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mstart,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_alt_l, show_coordinates=True, file=file + ".alt.l.256m.pdf",
            )

    if window_radius == 16000000:
        wpos = coord_clip(mend, chrlen)
        sequence = []
        for chrm, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
            seq = genome.get_encoding_from_coords(chrm, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1]
            else:
                seq = seq[None, :, :]
            sequence.append(seq)
        sequence = np.concatenate(sequence, axis=1)

    if mstart > wpos - window_radius:
        anno_scaled = process_anno(
            [[mstart, mend, "gray"]], base=wpos - window_radius, window_radius=window_radius,
        )
    else:
        anno_scaled = process_anno(
            [[wpos - window_radius, mend, "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    if window_radius == 16000000:
        outputs_alt_r = genomepredict(
            sequence, mchr, mend, wpos, models=models, annotation=anno_scaled, use_cuda=use_cuda
        )
        if file is not None:
            genomeplot(outputs_alt_r, show_coordinates=True, file=file + ".alt.r.pdf")
    else:
        outputs_alt_r = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mend,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_alt_r, show_coordinates=True, file=file + ".alt.r.256m.pdf",
            )

    return outputs_ref_l, outputs_ref_r, outputs_alt_l, outputs_alt_r


def process_ins(
    mchr,
    mpos,
    ins_seq,
    genome,
    strand="+",
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=16000000,
    padding_chr="chr1",
    use_cuda=True,
):
    """
    Generate multiscale genome interaction predictions for 
    an insertion variant that inserts the specified sequence 
    to the insertion site. 

    Parameters
    ----------
    mchr : str
        The chromosome name of the first segment
    mpos : int
        The insertion site coordinate.
    ins_seq : str
        The inserted sequence in string format.
    genome : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        The reference genome object to extract sequence from
    custom_models : list(torch.nn.Module or str) or None, optional
        Models to use instead of the default H1-ESC and HFF Orca models.
        Default is None.
    target : list(selene_utils2.Genomic2DFeatures or str) or bool, optional
        If specified as list, use this list of targets to retrieve experimental
        data (for plotting only). Default is True and will use micro-C data 
        for H1-ESC and HFF cells (4DNFI9GMP2J8, 4DNFI643OYP9) that correspond
        to the default models.
    file : str or None, optional
        Default is None. The output file prefix.
    show_genes : bool, optional
        Default is True. If True, generate gene annotation visualization
        file in pdf format that matches the windows of multiscale predictions.
    show_tracks : bool, optional
        Default is False. If True, generate chromatin tracks visualization
        file in pdf format that matches the windows of multiscale predictions.
    window_radius : int, optional
        Default is 16000000. The acceptable values are 16000000 which selects
        the 1-32Mb models or 128000000 which selects the 32-256Mb models.
    padding_chr : str, optional
        Default is "chr1". If window_radius is 128000000, padding is generally 
        needed to fill the sequence to 256Mb. The padding sequence will be 
        extracted from the padding_chr.
    use_cuda : bool, optional
        Default is True. Use CPU if False.

    Returns
    -------
    outputs_ref, outputs_alt_l, outputs_alt_r : dict, dict, dict
        Reference allele predictions zooming into the insertion site,
        Alternative allele predictions zooming into the left boundary of 
        the insertion seqeunce,
        Alternative allele prediction zooming into the right boundary of 
        the insertion seqeunce.
        The returned results are in the format of dictonaries 
        containing the prediction outputs and other 
        retrieved information. These dictionaries can be directly used as
        input to genomeplot or genomeplot_256Mb. See documentation of `genomepredict` or `genomepredict_256Mb` for
        details of the dictionary content.
    """
    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    if custom_models is None:
        if window_radius == 16000000:
            models = ["h1esc", "hff"]
        elif window_radius == 128000000:
            models = ["h1esc_256m", "hff_256m"]
        else:
            raise ValueError(
                "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
            )
    else:
        models = custom_models

    if target:
        try:
            if target == True:
                if window_radius == 16000000:
                    target = ["h1esc", "hff"]
                elif window_radius == 128000000:
                    target = ["h1esc_256m", "hff_256m"]
            target = [t if isinstance(t, Genomic2DFeatures) else target_dict_global[t] for t in target]
        except KeyError:
            target = False

    if window_radius == 16000000:
        wpos = coord_clip(mpos, chrlen)
        sequence = genome.get_encoding_from_coords(
            mchr, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        "chr" + mchr.replace("chr", ""),
                        coord_round(wpos - window_radius),
                        coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None
    elif window_radius == 128000000:
        chrlen_round = chrlen - chrlen % 32000
        wpos = 128000000
        if target:
            sequence, normmats, targets = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
        else:
            sequence, normmats = _retrieve_multi(
                [[mchr, 0, chrlen_round, "+"], [padding_chr, 0, 256000000 - chrlen_round, "+"]],
                genome,
                target=target,
            )
            targets = None
    else:
        raise ValueError(
            "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
        )

    anno_scaled = process_anno(
        [[mpos, "single"]], base=wpos - window_radius, window_radius=window_radius
    )

    if window_radius == 128000000:
        outputs_ref_l = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_round,
            mpos,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
    else:
        outputs_ref = genomepredict(
            sequence, mchr, mpos, wpos, annotation=anno_scaled, models=models, targets=targets, use_cuda=use_cuda,
        )

    if file is not None:
        if window_radius == 128000000:
            genomeplot_256Mb(
                outputs_ref_l, show_coordinates=True, file=file + ".ref.256m.pdf",
            )
        else:
            genomeplot(
                outputs_ref,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref.pdf",
            )

    # alt
    s = StructuralChange2(mchr, chrlen)
    s.insert(mpos, len(ins_seq), strand=strand)
    chrlen_alt = chrlen + len(ins_seq)
    if window_radius == 16000000:
        wpos = coord_clip(mpos, chrlen_alt)
        sequence = []
        for chr_name, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
            if chr_name.startswith("ins"):
                seq = Genome.sequence_to_encoding(ins_seq[start:end])
            else:
                seq = genome.get_encoding_from_coords(chr_name, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1]
            else:
                seq = seq[None, :, :]
            sequence.append(seq)
        sequence = np.concatenate(sequence, axis=1)
    else:
        chrlen_alt_round = chrlen_alt - chrlen_alt % 32000
        if chrlen_alt_round < 256000000:
            wpos = 128000000
            (sequence, normmats) = _retrieve_multi(
                list(s[0:chrlen_alt_round]) + [[padding_chr, 0, 256000000 - chrlen_alt_round, "+"]],
                genome,
                target=False,
                normmat=True,
                normmat_regionlist=[
                    [mchr, 0, chrlen_alt_round, "+"],
                    [padding_chr, 0, 256000000 - chrlen_alt_round, "+"],
                ],
            )
        else:
            wpos = coord_clip(mpos, chrlen_alt_round, window_radius=128000000)
            (sequence, normmats) = _retrieve_multi(
                list(s[wpos - window_radius : wpos + window_radius]),
                genome,
                target=False,
                normmat=True,
                normmat_regionlist=[[mchr, wpos - window_radius, wpos + window_radius, "+"]],
            )

    if mpos + len(ins_seq) < wpos + window_radius:
        anno_scaled = process_anno(
            [[mpos, mpos + len(ins_seq), "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    else:
        anno_scaled = process_anno(
            [[mpos, wpos + window_radius, "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 16000000:
        outputs_alt_l = genomepredict(
            sequence, mchr, mpos, wpos, models=models, annotation=anno_scaled, use_cuda=use_cuda
        )
        if file is not None:
            genomeplot(outputs_alt_l, show_coordinates=True, file=file + ".alt.l.pdf")
    else:
        outputs_alt_l = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_alt_round,
            mpos,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_alt_l, show_coordinates=True, file=file + ".alt.l.256m.pdf",
            )

    if window_radius == 16000000:
        wpos = coord_clip(mpos + len(ins_seq), chrlen_alt)
        sequence = []
        for chr_name, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
            if chr_name.startswith("ins"):
                seq = Genome.sequence_to_encoding(ins_seq[start:end])
            else:
                seq = genome.get_encoding_from_coords(chr_name, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1]
            else:
                seq = seq[None, :, :]
            sequence.append(seq)
        sequence = np.concatenate(sequence, axis=1)
    else:
        if chrlen_alt_round > 256000000:
            wpos = coord_clip(mpos + len(ins_seq), chrlen_alt_round, window_radius=128000000)
            (sequence, normmats) = _retrieve_multi(
                list(s[wpos - window_radius : wpos + window_radius]),
                genome,
                target=False,
                normmat=True,
                normmat_regionlist=[[mchr, wpos - window_radius, wpos + window_radius, "+"]],
            )

    if mpos > wpos - window_radius:
        anno_scaled = process_anno(
            [[mpos, mpos + len(ins_seq), "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )
    else:
        anno_scaled = process_anno(
            [[wpos - window_radius, mpos + len(ins_seq), "gray"]],
            base=wpos - window_radius,
            window_radius=window_radius,
        )

    if window_radius == 16000000:
        outputs_alt_r = genomepredict(
            sequence, mchr, mpos + len(ins_seq), wpos, annotation=anno_scaled, use_cuda=use_cuda
        )
        if file is not None:
            genomeplot(outputs_alt_r, show_coordinates=True, file=file + ".alt.r.pdf")
    else:
        outputs_alt = genomepredict_256Mb(
            sequence,
            mchr,
            normmats,
            chrlen_alt_round,
            mpos + len(ins_seq),
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_alt, show_coordinates=True, file=file + ".alt.r.256m.pdf",
            )

    return outputs_ref, outputs_alt_l, outputs_alt_r


def process_custom(
    region_list,
    ref_region_list,
    mpos,
    genome,
    ref_mpos_list=None,
    anno_list=None,
    ref_anno_list=None,
    custom_models=None,
    target=True,
    file=None,
    show_genes=True,
    show_tracks=False,
    window_radius=16000000,
    use_cuda=True,
):
    """
    Generate multiscale genome interaction predictions for 
    a custom variant by an ordered list of genomic segments.

    Parameters
    ----------
    region_list : list(list(...))
        List of segments to complete the alternative. Each segment is specified
        by a list( chr: str, start: int, end: int, strand: str), and segments
        are concatenated together in the given order. The total length 
        should sum up to 32Mb. An example input is
        [['chr5', 89411065, 89411065+16000000, '-'], ['chr7', 94378248, 94378248+16000000,'+']].
    ref_region_list : list(list(...))
        The reference regions to predict. This can be any reference regions with
        the length of the specified window size. If the Each reference region is specified
        with a list( chr: str, start: int, end: int, strand: str). The strand must
        be '+'. The intended use is predicting the genome interactions for each 
        segment that constitute the alternative allele within the native
        reference sequence context. An example
        input is [['chr5', 89411065-16000000, 89411065+16000000,'+'],
        ['chr7', 94378248-16000000, 94378248+16000000,'+']].
    mpos : int
        The position to zoom into in the alternative allele. Note that `mpos`
        here specify the relative position with respect to the to start of the 32Mb.
    genome : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        The reference genome object to extract sequence from.
    ref_mpos_list : list(int) or None, optional
        Default is None. List of positions to zoom into for each of the 
        reference regions specified in `ref_region_list`. If not specified, 
        then zoom into the center of each region. Note that `ref_mpos_list`
        specifies the relative positions with respect to start of the 32Mb. 
        For example, `16000000` means the center of the sequence.
    custom_models : list(torch.nn.Module or str) or None, optional
        Models to use instead of the default H1-ESC and HFF Orca models.
        Default is None.
    target : list(selene_utils2.Genomic2DFeatures or str) or bool, optional
        If specified as list, use this list of targets to retrieve experimental
        data (for plotting only). Default is True and will use micro-C data 
        for H1-ESC and HFF cells (4DNFI9GMP2J8, 4DNFI643OYP9) that correspond
        to the default models.
    file : str or None, optional
        Default is None. The output file prefix.
    show_genes : bool, optional
        Default is True. If True, generate gene annotation visualization
        file in pdf format that matches the windows of multiscale predictions.
    show_tracks : bool, optional
        Default is False. If True, generate chromatin tracks visualization
        file in pdf format that matches the windows of multiscale predictions.
    window_radius : int, optional
        Default is 16000000. Currently only 16000000 (32Mb window) is accepted.
    use_cuda : bool, optional
        Default is True. Use CPU if False.

    Returns
    -------
    outputs_ref_l, outputs_ref_r, outputs_alt : dict, dict, dict
        Reference allele predictions zooming into the left boundary of the
        duplication,
        Reference allele predictions zooming into the right boundary of the
        duplication,
        Alternative allele predictions zooming into the duplication breakpoint.
        The returned results are in the format of dictonaries 
        containing the prediction outputs and other 
        retrieved information. These dictionaries can be directly used as
        input to genomeplot or genomeplot_256Mb. See documentation of `genomepredict` or `genomepredict_256Mb` for
        details of the dictionary content.
        
    """
    if custom_models is None:
        if window_radius == 16000000:
            models = ["h1esc", "hff"]
        elif window_radius == 128000000:
            models = ["h1esc_256m", "hff_256m"]
        else:
            raise ValueError(
                "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
            )
    else:
        models = custom_models

    if target:
        try:
            if target == True:
                if window_radius == 16000000:
                    target = ["h1esc", "hff"]
                elif window_radius == 128000000:
                    target = ["h1esc_256m", "hff_256m"]
            target = [t if isinstance(t, Genomic2DFeatures) else target_dict_global[t] for t in target]
        except KeyError:
            target = False

    def validate_region_list(region_list, enforce_strand=None):
        sumlen = 0
        for chrm, start, end, strand in region_list:
            chrlen = [l for c, l in genome.get_chr_lens() if c == chrm].pop()
            assert start >= 0 and end <= chrlen
            sumlen += end - start
            if enforce_strand:
                if strand != enforce_strand:
                    raise ValueError("The specified strand must be " + enforce_strand)
        assert sumlen == 2 * window_radius

    validate_region_list(region_list)

    for i, ref_region in enumerate(ref_region_list):
        validate_region_list([ref_region], enforce_strand="+")
        ref_sequence = genome.get_encoding_from_coords(*ref_region)[None, :]

        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        ref_region[0], coord_round(ref_region[1]), coord_round(ref_region[2]),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None

        anno_scaled = process_anno(ref_anno_list, base=0, window_radius=window_radius)

        outputs_ref = genomepredict(
            ref_sequence,
            ref_region[0],
            ref_region[1] + window_radius if ref_mpos_list is None else ref_mpos_list[i],
            ref_region[1] + window_radius,
            annotation=anno_scaled,
            models=models,
            targets=targets,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot(
                outputs_ref,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref." + str(i) + ".pdf",
            )

    sequence = []
    for chrm, start, end, strand in region_list:
        seq = genome.get_encoding_from_coords(chrm, start, end)
        if strand == "-":
            seq = seq[None, ::-1, ::-1].copy()
        else:
            seq = seq[None, :, :]
        sequence.append(seq)
    alt_sequence = np.concatenate(sequence, axis=1)

    anno_scaled = process_anno(anno_list, base=0, window_radius=window_radius)

    outputs_alt = genomepredict(
        alt_sequence, "chimeric", mpos, window_radius, models=models, annotation=anno_scaled, use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(outputs_alt, show_coordinates=False, file=file + ".alt.pdf")
    return outputs_ref, outputs_alt


def process_single_breakpoint(
    chr1,
    pos1,
    chr2,
    pos2,
    orientation1,
    orientation2,
    genome,
    custom_models=None,
    target=True,
    file=None,
    show_genes=True,
    show_tracks=False,
    window_radius=16000000,
    padding_chr="chr1",
    use_cuda=True,
):
    """
    Generate multiscale genome interaction predictions for
    a simple translocation event that connects 
    two chromosomal breakpoints. Specifically, two breakpoint 
    positions and the corresponding two orientations are needed. 
    The orientations decide how the breakpoints are connected. 
    The ‘+’ or ‘-’ sign indicate whether the left or right side of 
    the breakpoint is used. For example, for an input 
    ('chr1', 85691449, 'chr5', 89533745 '+', '+'), two plus signs
    indicate connecting chr1:0-85691449 with chr5:0-89533745.

    Parameters
    ----------
    chr1 : str
        The chromosome name of the first segment
    pos1 : int
        The coorindate of breakpoint on the first segment
    chr2 : str
        The chromosome name of the second segment
    pos2 : int
        The coorindate of breakpoint on the second segment
    orientation1 : str
        Indicate which side of the breakpoint should be used for
        the first segment,
        '+' indicate the left and '-' indicate the right side.
    orientation2 : str
        Indicate which side of the breakpoint should be used for
        the second segment,
        '+' indicate the left and '-' indicate the right side.
    genome : selene_utils2.MemmapGenome or selene_sdk.sequences.Genome
        The reference genome object to extract sequence from
    custom_models : list(torch.nn.Module or str) or None, optional
        Models to use instead of the default H1-ESC and HFF Orca models.
        Default is None.
    target : list(selene_utils2.Genomic2DFeatures or str) or bool, optional
        If specified as list, use this list of targets to retrieve experimental
        data (for plotting only). Default is True and will use micro-C data 
        for H1-ESC and HFF cells (4DNFI9GMP2J8, 4DNFI643OYP9) that correspond
        to the default models.
    file : str or None, optional
        Default is None. The output file prefix.
    show_genes : bool, optional
        Default is True. If True, generate gene annotation visualization
        file in pdf format that matches the windows of multiscale predictions.
    show_tracks : bool, optional
        Default is False. If True, generate chromatin tracks visualization
        file in pdf format that matches the windows of multiscale predictions.
    window_radius : int, optional
        Default is 16000000. The acceptable values are 16000000 which selects
        the 1-32Mb models or 128000000 which selects the 32-256Mb models.
    padding_chr : str, optional
        Default is "chr1". If window_radius is 128000000, padding is generally 
        needed to fill the sequence to 256Mb. The padding sequence will be 
        extracted from the padding_chr.
    use_cuda : bool, optional
        Default is True. Use CPU if False.

    Returns
    -------
    outputs_ref_1, outputs_ref_2, outputs_alt : dict, dict, dict
        Reference allele predictions zooming into the chr1 breakpoint,
        Reference allele predictions zooming into the chr2 breakpoint,
        Alternative allele prediction zooming into the junction.
        The returned results are in the format of dictonaries 
        containing the prediction outputs and other 
        retrieved information. These dictionaries can be directly used as
        input to genomeplot or genomeplot_256Mb. See documentation of `genomepredict` or `genomepredict_256Mb` for
        details of the dictionary content.
    """

    if custom_models is None:
        if window_radius == 16000000:
            models = ["h1esc", "hff"]
        elif window_radius == 128000000:
            models = ["h1esc_256m", "hff_256m"]
        else:
            raise ValueError(
                "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
            )
    else:
        models = custom_models

    if target:
        try:
            if target == True:
                if window_radius == 16000000:
                    target = ["h1esc", "hff"]
                elif window_radius == 128000000:
                    target = ["h1esc_256m", "hff_256m"]
            target = [t if isinstance(t, Genomic2DFeatures) else target_dict_global[t] for t in target]
        except KeyError:
            target = False

    chrlen1 = [l for c, l in genome.get_chr_lens() if c == chr1].pop()
    # ref.l
    if window_radius == 16000000:
        wpos = coord_clip(pos1, chrlen1)
        sequence = genome.get_encoding_from_coords(
            chr1, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        chr1, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None
    elif window_radius == 128000000:
        chrlen1_round = chrlen1 - chrlen1 % 32000
        wpos = 128000000
        if target:
            sequence, normmats, targets = _retrieve_multi(
                [[chr1, 0, chrlen1_round, "+"], [padding_chr, 0, 256000000 - chrlen1_round, "+"]],
                genome,
                target=target,
            )
        else:
            sequence, normmats = _retrieve_multi(
                [[chr1, 0, chrlen1_round, "+"], [padding_chr, 0, 256000000 - chrlen1_round, "+"]],
                genome,
                target=target,
            )
            targets = None
    else:
        raise ValueError(
            "Only window_radius 16000000 (32Mb models) or 128000000 (256Mb models) are supported"
        )

    anno_scaled = process_anno(
        [[pos1, "single"]], base=wpos - window_radius, window_radius=window_radius
    )

    if window_radius == 128000000:
        outputs_ref_1 = genomepredict_256Mb(
            sequence,
            chr1,
            normmats,
            chrlen1_round,
            pos1,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
    else:
        outputs_ref_1 = genomepredict(
            sequence, chr1, pos1, wpos, models=models, annotation=anno_scaled, targets=targets, use_cuda=use_cuda,
        )
    if file is not None:
        if window_radius == 128000000:
            genomeplot_256Mb(
                outputs_ref_1, show_coordinates=True, file=file + ".ref.1.256m.pdf",
            )
        else:
            genomeplot(
                outputs_ref_1,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref.1.pdf",
                colorbar=True,
            )

    chrlen2 = [l for c, l in genome.get_chr_lens() if c == chr2].pop()
    if window_radius == 16000000:
        wpos = coord_clip(pos2, chrlen2)

        sequence = genome.get_encoding_from_coords(
            chr2, wpos - window_radius, wpos + window_radius
        )[None, :]
        if target:
            targets = [
                torch.FloatTensor(
                    t.get_feature_data(
                        chr2, coord_round(wpos - window_radius), coord_round(wpos + window_radius),
                    )[None, :]
                ) for t in target
            ]
        else:
            targets = None
    elif window_radius == 128000000:
        chrlen2_round = chrlen2 - chrlen2 % 32000
        wpos = 128000000
        if target:
            sequence, normmats, targets = _retrieve_multi(
                [[chr2, 0, chrlen2_round, "+"], [padding_chr, 0, 256000000 - chrlen2_round, "+"]],
                genome,
                target=target,
            )
        else:
            sequence, normmats = _retrieve_multi(
                [[chr2, 0, chrlen2_round, "+"], [padding_chr, 0, 256000000 - chrlen2_round, "+"]],
                genome,
                target=target,
            )
            targets = None

    anno_scaled = process_anno(
        [[pos2, "single"]], base=wpos - window_radius, window_radius=window_radius
    )

    if window_radius == 128000000:
        outputs_ref_2 = genomepredict_256Mb(
            sequence,
            chr2,
            normmats,
            chrlen2_round,
            pos2,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            targets=targets,
            use_cuda=use_cuda,
        )
    else:
        outputs_ref_2 = genomepredict(
            sequence, chr2, pos2, wpos, models=models, annotation=anno_scaled, targets=targets, use_cuda=use_cuda,
        )

    if file is not None:
        if window_radius == 128000000:
            genomeplot_256Mb(
                outputs_ref_2, show_coordinates=True, file=file + ".ref.2.256m.pdf",
            )
        else:
            genomeplot(
                outputs_ref_2,
                show_genes=show_genes,
                show_tracks=show_tracks,
                show_coordinates=True,
                file=file + ".ref.2.pdf",
                colorbar=True,
            )

    chrlen = [l for c, l in genome.get_chr_lens() if c == chr1].pop()
    s = StructuralChange2(chr1, chrlen)
    if orientation1 == "+":
        s.delete(pos1, chrlen)
    else:
        s.delete(0, pos1 - 1)
        s.invert(0, chrlen - pos1 + 1)

    chrlen = [l for c, l in genome.get_chr_lens() if c == chr2].pop()
    s2 = StructuralChange2(chr2, chrlen)
    if orientation2 == "-":
        s2.delete(0, pos2 - 1)
    else:
        s2.delete(pos2, chrlen)
        s2.invert(0, pos2)

    breakpos = s.coord_points[-1]
    s = s + s2

    if window_radius == 16000000:
        wpos = coord_clip(breakpos, s.coord_points[-1])

        sequence = []
        curpos = 0
        anno = []
        for chrm, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
            seq = genome.get_encoding_from_coords(chrm, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1]
            else:
                seq = seq[None, :, :]
            sequence.append(seq)
            anno.append([curpos, curpos + end - start])
            curpos = curpos + end - start
        sequence = np.concatenate(sequence, axis=1)
    else:
        chrlen_alt_round = s.coord_points[-1] - s.coord_points[-1] % 32000
        if chrlen_alt_round < 256000000:
            wpos = 128000000
            (sequence, normmats) = _retrieve_multi(
                list(s[0:chrlen_alt_round]) + [[padding_chr, 0, 256000000 - chrlen_alt_round, "+"]],
                genome,
                target=False,
                normmat=True,
                normmat_regionlist=[
                    [chr1 + "|" + chr2, 0, chrlen_alt_round, "+"],
                    [padding_chr, 0, 256000000 - chrlen_alt_round, "+"],
                ],
            )
            curpos = 0
            anno = []
            for chrm, start, end, strand in s[0:chrlen_alt_round]:
                anno.append([curpos, curpos + end - start])
                curpos = curpos + end - start
        else:
            wpos = coord_clip(breakpos, chrlen_alt_round, window_radius=128000000)
            (sequence, normmats) = _retrieve_multi(
                list(s[wpos - window_radius : wpos + window_radius]),
                genome,
                target=False,
                normmat=True,
                normmat_regionlist=[
                    [chr1 + "|" + chr2, wpos - window_radius, wpos + window_radius, "+"]
                ],
            )
            curpos = 0
            anno = []
            for chrm, start, end, strand in s[wpos - window_radius : wpos + window_radius]:
                anno.append([curpos, curpos + end - start])
                curpos = curpos + end - start

    anno_scaled = process_anno([[anno[0][-1], "double"]], base=0, window_radius=window_radius)
    if window_radius == 16000000:
        outputs_alt = genomepredict(
            sequence, chr1 + "|" + chr2, breakpos, wpos, models=models, annotation=anno_scaled, use_cuda=use_cuda
        )
        if file is not None:
            genomeplot(outputs_alt, show_coordinates=False, file=file + ".alt.pdf", colorbar=True)
    else:
        outputs_alt = genomepredict_256Mb(
            sequence,
            chr1 + "|" + chr2,
            normmats,
            chrlen_alt_round,
            breakpos,
            wpos,
            models=models,
            annotation=anno_scaled,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
        )
        if file is not None:
            genomeplot_256Mb(
                outputs_alt, show_coordinates=True, file=file + ".alt.256m.pdf",
            )
    return outputs_ref_1, outputs_ref_2, outputs_alt


if __name__ == "__main__":
    from docopt import docopt
    import sys
    import os
    import re

    doc = """
    Orca multiscale genome interaction sequence model prediction tool.

    Usage:
    orca_predict region [options] <coordinate> <output_dir>
    orca_predict del [options] <coordinate> <output_dir>
    orca_predict dup [options] <coordinate> <output_dir>
    orca_predict inv [options] <coordinate> <output_dir>
    orca_predict break [options] <coordinate> <output_dir>

    Options:
    -h --help        Show this screen.
    --show_genes     Show gene annotation (only supported for 32Mb models).
    --show_tracks    Show chromatin tracks (only supported for 32Mb models).
    --256m           Use 256Mb models (default is 32Mb).
    --nocuda         Use CPU implementation.
    --coor_filename  Include coordinate in the output filenames.
    --version        Show version.
    """
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    arguments = docopt(doc, version="Orca v0.1")
    show_genes = arguments["--show_genes"]
    show_tracks = arguments["--show_tracks"]
    window_radius = 128000000 if arguments["--256m"] else 16000000
    use_cuda = not arguments["--nocuda"]
    coor_filename = arguments["--coor_filename"]

    load_resources(models=["256M" if arguments["--256m"] else "32M"], use_cuda=use_cuda)

    if arguments["region"]:
        predtype = "region"
    elif arguments["del"]:
        predtype = "del"
    elif arguments["dup"]:
        predtype = "dup"
    elif arguments["inv"]:
        predtype = "inv"
    elif arguments["break"]:
        predtype = "break"
    
    if coor_filename:
        suffix = "_" + re.sub(r'[\\/*?:"<>|]', "_", arguments["<coordinate>"])
    else:
        suffix = ""
    
    def predict(chrm, start, end, savedir):

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        with torch.no_grad():
            outputs = process_region(
                chrm,
                start,
                end,
                hg38,
                target=target_available,
                file=savedir + "/orca_pred" + suffix,
                show_genes=show_genes,
                show_tracks=show_tracks,
                window_radius=window_radius,
                padding_chr="chr1",
                use_cuda=use_cuda,
            )
        torch.save(outputs, savedir + "/orca_pred" + suffix+ ".pth")
        return None

    def get_interactions(predtype, content, savedir):

        if predtype == "region":
            pdf_names = ["orca_pred" + suffix+ ".pdf"]
            if show_genes or show_tracks:
                pdf_names += ["orca_pred" + suffix+ ".anno.pdf"]
            chrstr, coordstr = str(content).split(":")
            chrstr = "chr" + chrstr.replace("chr", "")
            coord_s, coord_e = coordstr.split("-")
            predict(chrstr, int(coord_s), int(coord_e), savedir)
        elif predtype in ["dup", "del"]:
            pdf_names = ["orca_pred" + suffix + ".ref.l.pdf", "orca_pred" + suffix + ".ref.r.pdf", "orca_pred" + suffix + ".alt.pdf"]
            if show_genes or show_tracks:
                pdf_names += [
                    "orca_pred" + suffix + "d.ref.l.anno.pdf",
                    "orca_pred" + suffix + ".ref.r.anno.pdf",
                    "orca_pred" + suffix + ".alt.anno.pdf",
                ]
            chrstr, coordstr = str(content).split(":")
            chrstr = "chr" + chrstr.replace("chr", "")
            coord_s, coord_e = coordstr.split("-")

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            if predtype == "dup":
                outputs_ref_l, outputs_ref_r, outputs_alt = process_dup(
                    chrstr,
                    int(coord_s),
                    int(coord_e),
                    hg38,
                    target=target_available,
                    show_genes=show_genes,
                    show_tracks=show_tracks,
                    file=savedir + "/orca_pred" + suffix,
                    window_radius=window_radius,
                    use_cuda=use_cuda,
                )
            else:
                outputs_ref_l, outputs_ref_r, outputs_alt = process_del(
                    chrstr,
                    int(coord_s),
                    int(coord_e),
                    hg38,
                    target=target_available,
                    show_genes=show_genes,
                    show_tracks=show_tracks,
                    file=savedir + "/orca_pred" + suffix,
                    window_radius=window_radius,
                    use_cuda=use_cuda,
                )
            torch.save(
                {
                    "outputs_ref_l": outputs_ref_l,
                    "outputs_ref_r": outputs_ref_r,
                    "outputs_alt": outputs_alt,
                },
                savedir + "/orca_pred.pth",
            )
        elif predtype == "inv":
            pdf_names = [
                "orca_pred" + suffix + ".ref.l.pdf",
                "orca_pred" + suffix + ".ref.r.pdf",
                "orca_pred" + suffix + ".alt.l.pdf",
                "orca_pred" + suffix + ".alt.r.pdf",
            ]
            if show_genes or show_tracks:
                pdf_names += [
                    "orca_pred" + suffix + ".ref.l.anno.pdf",
                    "orca_pred" + suffix + ".ref.r.anno.pdf",
                    "orca_pred" + suffix + ".alt.l.anno.pdf",
                    "orca_pred" + suffix + ".alt.r.anno.pdf",
                ]
            chrstr, coordstr = str(content).split(":")
            chrstr = "chr" + chrstr.replace("chr", "")
            coord_s, coord_e = coordstr.split("-")

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            outputs_ref_l, outputs_ref_r, outputs_alt_l, outputs_alt_r = process_inv(
                chrstr,
                int(coord_s),
                int(coord_e),
                hg38,
                target=target_available,
                show_genes=show_genes,
                show_tracks=show_tracks,
                file=savedir + "/orca_pred" + suffix,
                window_radius=window_radius,
                use_cuda=use_cuda,
            )

            torch.save(
                {
                    "outputs_ref_l": outputs_ref_l,
                    "outputs_ref_r": outputs_ref_r,
                    "outputs_alt_l": outputs_alt_l,
                    "outputs_alt_r": outputs_alt_r,
                },
                savedir + "/orca_pred" + suffix + ".pth",
            )
        elif predtype == "break":
            pdf_names = ["orca_pred.ref.1.pdf", "orca_pred.ref.2.pdf", "orca_pred.alt.pdf"]
            if show_genes or show_tracks:
                pdf_names += [
                    "orca_pred" + suffix + ".ref.1.anno.pdf",
                    "orca_pred" + suffix + ".ref.2.anno.pdf",
                    "orca_pred" + suffix + ".alt.anno.pdf",
                ]
            chr_coord_1, chr_coord_2, orientations = str(content.replace("\t", " ")).split(" ")
            chr1, coord1 = chr_coord_1.split(":")
            chr2, coord2 = chr_coord_2.split(":")
            chr1 = "chr" + chr1.replace("chr", "")
            chr2 = "chr" + chr2.replace("chr", "")
            orientation1, orientation2 = orientations.split("/")

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            outputs_ref_1, outputs_ref_2, outputs_alt = process_single_breakpoint(
                chr1,
                int(coord1),
                chr2,
                int(coord2),
                orientation1,
                orientation2,
                hg38,
                target=target_available,
                show_genes=show_genes,
                show_tracks=show_tracks,
                file=savedir + "/orca_pred" + suffix,
                window_radius=window_radius,
                use_cuda=use_cuda,
            )

            torch.save(
                {
                    "outputs_ref_1": outputs_ref_1,
                    "outputs_ref_2": outputs_ref_2,
                    "outputs_alt": outputs_alt,
                },
                savedir + "/orca_pred" + suffix + ".pth",
            )
        else:
            raise ValueError("Unexpected prediction type!")
        return None

    get_interactions(predtype, arguments["<coordinate>"], arguments["<output_dir>"])

