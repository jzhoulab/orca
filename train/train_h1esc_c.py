import sys
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import selene_sdk
from selene_sdk.samplers.dataloader import SamplerDataLoader

sys.path.append("..")
from selene_utils2 import *
from orca_modules import Decoder, Decoder_1m, Encoder, Encoder2, Encoder3

modelstr = "h1esc_c"
seed = 3141

torch.set_default_tensor_type("torch.FloatTensor")
os.makedirs("./models/", exist_ok=True)
os.makedirs("./png/", exist_ok=True)

MODELA_PATH = "./models/model_h1esc_a_swa.checkpoint"
MODELB_PATH = "./models/model_h1esc_b.checkpoint"
if __name__ == "__main__":

    t = Genomic2DFeatures(
        ["../resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/32000"],
        ["r32000"],
        (8000, 8000),
        cg=True,
    )
    sampler = RandomPositionsSamplerHiC(
        reference_sequence=MemmapGenome(
            "../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
            memmapfile="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
        ),
        target=t,
        features=["r4000"],
        test_holdout=["chr9", "chr10"],
        validation_holdout=["chr8"],
        sequence_length=256000000,
        position_resolution=32000,
        random_shift=4000,
        random_strand=True,
        cross_chromosome=True,
        permute_segments=True,
        background_cis_file="../resources/4DNFI9GMP2J8.rebinned.mcool.expected.res32000.mono.npy",
        background_trans_file="../resources/4DNFI9GMP2J8.rebinned.mcool.expected.res32000.trans.npy",
    )

    sampler.mode = "train"
    dataloader = SamplerDataLoader(sampler, num_workers=8, batch_size=1, seed=seed)

    def figshow(x, np=False):
        if np:
            plt.imshow(x.squeeze())
        else:
            plt.imshow(x.squeeze().cpu().detach().numpy())
        plt.show()

    try:
        net = nn.DataParallel(Encoder3())
        denet_32 = nn.DataParallel(Decoder())
        denet_64 = nn.DataParallel(Decoder())
        denet_128 = nn.DataParallel(Decoder())
        denet_256 = nn.DataParallel(Decoder())
        net.load_state_dict(torch.load("./models/model_" + modelstr + ".checkpoint"))
        denet_32.load_state_dict(torch.load("./models/model_" + modelstr + ".d32.checkpoint"))
        denet_64.load_state_dict(torch.load("./models/model_" + modelstr + ".d64.checkpoint"))
        denet_128.load_state_dict(torch.load("./models/model_" + modelstr + ".d128.checkpoint"))
        denet_256.load_state_dict(torch.load("./models/model_" + modelstr + ".d256.checkpoint"))
    except:
        print("no pretrained model found!")
        net = nn.DataParallel(Encoder3())
        denet_32 = nn.DataParallel(Decoder())
        denet_64 = nn.DataParallel(Decoder())
        denet_128 = nn.DataParallel(Decoder())
        denet_256 = nn.DataParallel(Decoder())

    net0 = nn.DataParallel(Encoder())
    net0_dict = net0.state_dict()
    pretrained_dict = torch.load(MODELA_PATH)
    pretrained_dict_filtered = {key: pretrained_dict[key] for key in net0_dict}
    net0.load_state_dict(pretrained_dict_filtered)
    for param in net0.parameters():
        param.requires_grad = False

    net1 = nn.DataParallel(Encoder2())
    net1_dict = net1.state_dict()
    pretrained_dict = torch.load(MODELB_PATH)
    pretrained_dict_filtered = {key: pretrained_dict[key] for key in net1_dict}
    net1.load_state_dict(pretrained_dict_filtered)
    for param in net1.parameters():
        param.requires_grad = False

    net0.cuda()
    net1.cuda()
    net.cuda()
    denet_32.cuda()
    denet_64.cuda()
    denet_128.cuda()
    denet_256.cuda()

    net0.eval()
    net1.eval()
    net.train()
    denet_32.train()
    denet_64.train()
    denet_128.train()
    denet_256.train()

    params = (
        [p for p in net.parameters() if p.requires_grad]
        + [p for p in denet_32.parameters() if p.requires_grad]
        + [p for p in denet_64.parameters() if p.requires_grad]
        + [p for p in denet_128.parameters() if p.requires_grad]
        + [p for p in denet_256.parameters() if p.requires_grad]
    )

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.98)
    try:
        optimizer_bak = torch.load("./models/model_" + modelstr + ".optimizer")
        optimizer.load_state_dict(optimizer_bak)
    except:
        print("no saved optimizer found!")

    i = 0
    denets = {32: denet_32, 64: denet_64, 128: denet_128, 256: denet_256}
    past_losses = {32: [], 64: [], 128: [], 256: []}

    sequence = torch.zeros((4, 256000000, 4), dtype=torch.float32)
    target = torch.zeros((4, 8000, 8000), dtype=torch.float32)
    normmat = np.zeros((4, 8000, 8000), dtype=np.float32)

    while True:
        fillind = 0
        for ss, tt, nn in dataloader:
            if np.isnan(tt.numpy()).mean() > 0.5:
                continue
            sequence[fillind, :] = ss
            target[fillind, :] = tt
            normmat[fillind, :] = nn
            if fillind == sequence.shape[0] - 1:
                fillind = 0
            else:
                fillind += 1
                continue

            optimizer.zero_grad()
            with torch.no_grad():
                encoding0 = net1(net0(torch.Tensor(sequence.float()).transpose(1, 2).cuda()))[-1]
            encoding32, encoding64, encoding128, encoding256 = net(encoding0)
            encodings = {32: encoding32, 64: encoding64, 128: encoding128, 256: encoding256}

            def train_step(level, start, coarse_pred=None):
                d = int(level / 8)
                target_r = np.nanmean(
                    np.reshape(
                        target[:, start : start + 250 * d, start : start + 250 * d].numpy(),
                        (target.shape[0], 250, d, 250, d),
                    ),
                    axis=(2, 4),
                )
                normmat_nan = np.isnan(normmat)
                if np.any(normmat_nan):
                    normmat[normmat_nan] = np.nanmin(normmat[~normmat_nan])
                normmat_r = np.mean(
                    np.mean(
                        np.reshape(
                            normmat[:, start : start + 250 * d, start : start + 250 * d],
                            (normmat.shape[0], 250, d, 250, d),
                        ),
                        axis=4,
                    ),
                    axis=2,
                )
                if coarse_pred is not None:
                    pred = denets[level].forward(
                        encodings[level][:, :, int(start / d) : int(start / d) + 250],
                        torch.log(torch.Tensor(normmat_r).cuda())[:, None, :, :],
                        coarse_pred,
                    )
                else:
                    pred = denets[level].forward(
                        encodings[level][:, :, int(start / d) : int(start / d) + 250],
                        torch.log(torch.Tensor(normmat_r).cuda())[:, None, :, :],
                    )

                eps = np.nanmin(normmat_r)
                target_cuda = torch.Tensor(
                    np.log(((eps + target_r) / (eps + normmat_r)))[:, 0:, 0:]
                ).cuda()
                loss = (
                    (
                        pred[:, 0, 0:, 0:][torch.isfinite(target_cuda)]
                        - target_cuda[torch.isfinite(target_cuda)]
                    )
                    ** 2
                ).sum() / (pred.shape[0] * 250 ** 2)
                loss = loss
                past_losses[level].append(loss.detach().cpu().numpy())
                return loss, pred

            start = 0
            loss256, pred = train_step(256, start)
            r = np.random.randint(0, 125)
            start = start + r * 32
            loss128, pred = train_step(128, start, pred[:, :, r : r + 125, r : r + 125].detach())
            r = np.random.randint(0, 125)
            start = start + r * 16
            loss64, pred = train_step(64, start, pred[:, :, r : r + 125, r : r + 125].detach())
            r = np.random.randint(0, 125)
            start = start + r * 8
            loss32, pred = train_step(32, start, pred[:, :, r : r + 125, r : r + 125].detach())

            loss = loss256 + loss128 + loss64 + loss32
            loss.backward()
            optimizer.step()

            del encodings
            del encoding32
            del encoding64
            del encoding128
            del encoding256
            del pred

            if i % 500 == 0:

                print("l32:" + str(np.mean(past_losses[32][-500:])), flush=True)
                print("l64:" + str(np.mean(past_losses[64][-500:])), flush=True)
                print("l128:" + str(np.mean(past_losses[128][-500:])), flush=True)
                print("l256:" + str(np.mean(past_losses[256][-500:])), flush=True)

            if i % 500 == 0:
                torch.save(net.state_dict(), "./models/model_" + modelstr + ".checkpoint")
                torch.save(denet_32.state_dict(), "./models/model_" + modelstr + ".d32.checkpoint")
                torch.save(denet_64.state_dict(), "./models/model_" + modelstr + ".d64.checkpoint")
                torch.save(
                    denet_128.state_dict(), "./models/model_" + modelstr + ".d128.checkpoint"
                )
                torch.save(
                    denet_256.state_dict(), "./models/model_" + modelstr + ".d256.checkpoint"
                )
                torch.save(optimizer.state_dict(), "./models/model_" + modelstr + ".optimizer")

            if i % 2000 == 0 and i != 0:
                t2 = Genomic2DFeatures(
                    ["../resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/32000"],
                    ["r32000"],
                    (8000, 8000),
                    cg=True,
                )
                t2.mode = "validate"
                sampler2 = RandomPositionsSamplerHiC(
                    reference_sequence=MemmapGenome(
                        "../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                        memmapfile="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
                    ),
                    target=t2,
                    features=["r32000"],
                    test_holdout=["chr9", "chr10"],
                    validation_holdout=["chr8"],
                    sequence_length=256000000,
                    position_resolution=32000,
                    random_shift=0,
                    random_strand=True,
                    cross_chromosome=True,
                    permute_segments=True,
                    background_cis_file="../resources/4DNFI9GMP2J8.rebinned.mcool.expected.res32000.mono.npy",
                    background_trans_file="../resources/4DNFI9GMP2J8.rebinned.mcool.expected.res32000.trans.npy",
                )

                sampler2.mode = "validate"
                dataloader2 = SamplerDataLoader(sampler2, num_workers=4, batch_size=1)

                net.eval()
                denet_32.eval()
                denet_64.eval()
                denet_128.eval()
                denet_256.eval()

                count = 0
                mses = {32: [], 64: [], 128: [], 256: []}
                corrranks = {32: [], 64: [], 128: [], 256: []}

                fillind = 0
                for ss, tt, nn in dataloader2:
                    if np.isnan(tt.numpy()).mean() > 0.3:
                        continue
                    sequence[fillind, :] = ss
                    target[fillind, :] = tt
                    normmat[fillind, :] = nn
                    if fillind == sequence.shape[0] - 1:
                        fillind = 0
                    else:
                        fillind += 1
                        continue

                    count += 1
                    if count == 100:
                        break
                    with torch.no_grad():
                        encoding32, encoding64, encoding128, encoding256 = net(
                            net1(net0(torch.Tensor(sequence.float()).transpose(1, 2).cuda()))[-1]
                        )

                    encodings = {32: encoding32, 64: encoding64, 128: encoding128, 256: encoding256}

                    def eval_step(level, start, coarse_pred=None, plot=False):
                        global normmat
                        d = int(level / 8)
                        target_r = np.nanmean(
                            np.nanmean(
                                np.reshape(
                                    target[
                                        :, start : start + 250 * d, start : start + 250 * d
                                    ].numpy(),
                                    (target.shape[0], 250, d, 250, d),
                                ),
                                axis=4,
                            ),
                            axis=2,
                        )
                        normmat_nan = np.isnan(normmat)
                        if np.any(normmat_nan):
                            normmat[normmat_nan] = np.nanmin(normmat[~normmat_nan])
                        normmat_r = np.mean(
                            np.mean(
                                np.reshape(
                                    normmat[:, start : start + 250 * d, start : start + 250 * d],
                                    (normmat.shape[0], 250, d, 250, d),
                                ),
                                axis=4,
                            ),
                            axis=2,
                        )
                        with torch.no_grad():
                            if coarse_pred is not None:
                                pred = denets[level].forward(
                                    encodings[level][:, :, int(start / d) : int(start / d) + 250],
                                    torch.log(torch.Tensor(normmat_r).cuda())[:, None, :, :],
                                    coarse_pred,
                                )
                            else:
                                pred = denets[level].forward(
                                    encodings[level][:, :, int(start / d) : int(start / d) + 250],
                                    torch.log(torch.Tensor(normmat_r).cuda())[:, None, :, :],
                                )

                        eps = np.nanmin(normmat_r)
                        target_cuda = torch.Tensor(
                            np.log(((target_r + eps) / (normmat_r + eps)))[:, 0:, 0:]
                        ).cuda()
                        loss = (
                            (
                                pred[:, 0, 0:, 0:][torch.isfinite(target_cuda)]
                                - target_cuda[torch.isfinite(target_cuda)]
                            )
                            ** 2
                        ).sum() / (pred.shape[0] * 250 ** 2)

                        mses[level].append(loss.detach().cpu().numpy())
                        pred_np = (
                            pred[:, 0, 0:, 0:].detach().cpu().numpy().reshape((pred.shape[0], -1))
                        )
                        target_np = np.log((eps + target_r) / (eps + normmat_r))[:, 0:, 0:].reshape(
                            (pred.shape[0], -1)
                        )

                        for j in range(pred_np.shape[0]):
                            validinds = ~np.isnan(target_np[j, :])
                            corrranks[level].append(
                                pearsonr(pred_np[j, validinds], target_np[j, validinds])[0]
                            )
                        if plot:
                            for ii in range(pred.shape[0]):
                                figshow(pred[ii, 0, :, :])
                                plt.savefig(
                                    "./png/model_"
                                    + modelstr
                                    + ".test"
                                    + str(ii)
                                    + ".level"
                                    + str(level)
                                    + ".pred.png"
                                )
                                figshow(
                                    np.log((eps + target_r) / (eps + normmat_r))[ii, :, :], np=True
                                )
                                plt.savefig(
                                    "./png/model_"
                                    + modelstr
                                    + ".test"
                                    + str(ii)
                                    + ".level"
                                    + str(level)
                                    + ".label.png"
                                )
                        return pred

                    start = 0
                    pred = eval_step(256, start, plot=count == 99)
                    start = start + 62 * 32
                    pred = eval_step(128, start, pred[:, :, 62:187, 62:187], plot=count == 99)
                    start = start + 63 * 16
                    pred = eval_step(64, start, pred[:, :, 63:188, 63:188], plot=count == 99)
                    start = start + 63 * 8
                    pred = eval_step(32, start, pred[:, :, 63:188, 63:188], plot=count == 99)

                    del encodings
                    del encoding32
                    del encoding64
                    del encoding128
                    del encoding256
                    del pred

                print(
                    " Cor32 {0}, Cor64 {1}, Cor128 {2}, Cor256 {3}".format(
                        np.nanmean(corrranks[32]),
                        np.nanmean(corrranks[64]),
                        np.nanmean(corrranks[128]),
                        np.nanmean(corrranks[256]),
                    )
                )
                net.train()
                denet_32.train()
                denet_64.train()
                denet_128.train()
                denet_256.train()

            i += 1

