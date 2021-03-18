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
from orca_modules import Decoder, Decoder_1m, Encoder, Encoder2

modelstr = "hctnoc_b"
seed = 3141

torch.set_default_tensor_type("torch.FloatTensor")
os.makedirs("./models/", exist_ok=True)
os.makedirs("./png/", exist_ok=True)

MODELA_PATH = "./models/model_hctnoc_a_swa.checkpoint"
if __name__ == "__main__":

    smooth_diag = np.load(
        "../resources/4DNFILP99QJS.HCT_auxin6h.rebinned.mcool.expected.res4000.npy"
    )
    normmat = np.exp(smooth_diag[np.abs(np.arange(8000)[:, None] - np.arange(8000)[None, :])])

    t = Genomic2DFeatures(
        ["../resources/4DNFILP99QJS.HCT_auxin6h.rebinned.mcool::/resolutions/4000"],
        ["r4000"],
        (8000, 8000),
        cg=True,
    )
    sampler = RandomPositionsSamplerHiC(
        reference_sequence=MemmapGenome(
            input_path="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
            memmapfile="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
            blacklist_regions="hg38",
        ),
        target=t,
        features=["r4000"],
        test_holdout=["chr9", "chr10"],
        validation_holdout=["chr8"],
        sequence_length=32000000,
        position_resolution=4000,
        random_shift=1000,
        random_strand=True,
        cross_chromosome=False,
    )

    sampler.mode = "train"
    dataloader = SamplerDataLoader(sampler, num_workers=24, batch_size=1, seed=seed)

    def figshow(x, np=False):
        if np:
            plt.imshow(x.squeeze())
        else:
            plt.imshow(x.squeeze().cpu().detach().numpy())
        plt.show()

    try:
        net = nn.DataParallel(Encoder2())
        denet_1 = nn.DataParallel(Decoder())
        denet_2 = nn.DataParallel(Decoder())
        denet_4 = nn.DataParallel(Decoder())
        denet_8 = nn.DataParallel(Decoder())
        denet_16 = nn.DataParallel(Decoder())
        denet_32 = nn.DataParallel(Decoder())
        net.load_state_dict(torch.load("./models/model_" + modelstr + ".checkpoint"))
        denet_1.load_state_dict(torch.load("./models/model_" + modelstr + ".d1.checkpoint"))
        denet_2.load_state_dict(torch.load("./models/model_" + modelstr + ".d2.checkpoint"))
        denet_4.load_state_dict(torch.load("./models/model_" + modelstr + ".d4.checkpoint"))
        denet_8.load_state_dict(torch.load("./models/model_" + modelstr + ".d8.checkpoint"))
        denet_16.load_state_dict(torch.load("./models/model_" + modelstr + ".d16.checkpoint"))
        denet_32.load_state_dict(torch.load("./models/model_" + modelstr + ".d32.checkpoint"))
    except:
        print("no pretrained model found!")
        net = nn.DataParallel(Encoder2())
        denet_1 = nn.DataParallel(Decoder())
        denet_2 = nn.DataParallel(Decoder())
        denet_4 = nn.DataParallel(Decoder())
        denet_8 = nn.DataParallel(Decoder())
        denet_16 = nn.DataParallel(Decoder())
        denet_32 = nn.DataParallel(Decoder())

    net0 = nn.DataParallel(Encoder())
    net0_dict = net0.state_dict()
    pretrained_dict = torch.load(MODELA_PATH)
    pretrained_dict_filtered = {key: pretrained_dict[key] for key in net0_dict}
    net0.load_state_dict(pretrained_dict_filtered)
    for param in net0.parameters():
        param.requires_grad = False

    net0.cuda()
    net.cuda()
    denet_1.cuda()
    denet_2.cuda()
    denet_4.cuda()
    denet_8.cuda()
    denet_16.cuda()
    denet_32.cuda()

    net0.eval()
    net.train()
    denet_1.train()
    denet_2.train()
    denet_4.train()
    denet_8.train()
    denet_16.train()
    denet_32.train()

    params = (
        [p for p in net.parameters() if p.requires_grad]
        + [p for p in denet_1.parameters() if p.requires_grad]
        + [p for p in denet_2.parameters() if p.requires_grad]
        + [p for p in denet_4.parameters() if p.requires_grad]
        + [p for p in denet_8.parameters() if p.requires_grad]
        + [p for p in denet_16.parameters() if p.requires_grad]
        + [p for p in denet_32.parameters() if p.requires_grad]
    )

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.98)
    try:
        optimizer_bak = torch.load("./models/model_" + modelstr + ".optimizer")
        optimizer.load_state_dict(optimizer_bak)
    except:
        print("no saved optimizer found!")

    i = 0
    normmat_r1 = np.reshape(normmat[:250, :250], (250, 1, 250, 1)).mean(axis=1).mean(axis=2)
    normmat_r2 = np.reshape(normmat[:500, :500], (250, 2, 250, 2)).mean(axis=1).mean(axis=2)
    normmat_r4 = np.reshape(normmat[:1000, :1000], (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
    normmat_r8 = np.reshape(normmat[:2000, :2000], (250, 8, 250, 8)).mean(axis=1).mean(axis=2)
    normmat_r16 = np.reshape(normmat[:4000, :4000], (250, 16, 250, 16)).mean(axis=1).mean(axis=2)
    normmat_r32 = np.reshape(normmat[:8000, :8000], (250, 32, 250, 32)).mean(axis=1).mean(axis=2)
    eps1 = np.min(normmat_r1)
    eps2 = np.min(normmat_r2)
    eps4 = np.min(normmat_r4)
    eps8 = np.min(normmat_r8)
    eps16 = np.min(normmat_r16)
    eps32 = np.min(normmat_r32)

    normmats = {
        1: normmat_r1,
        2: normmat_r2,
        4: normmat_r4,
        8: normmat_r8,
        16: normmat_r16,
        32: normmat_r32,
    }
    epss = {1: eps1, 2: eps2, 4: eps4, 8: eps8, 16: eps16, 32: eps32}
    denets = {1: denet_1, 2: denet_2, 4: denet_4, 8: denet_8, 16: denet_16, 32: denet_32}

    past_losses = {1: [], 2: [], 4: [], 8: [], 16: [], 32: []}

    sequence = torch.zeros((4, 32000000, 4), dtype=torch.float32)
    target = torch.zeros((4, 8000, 8000), dtype=torch.float32)
    while True:
        fillind = 0
        optimizer.zero_grad()
        for ss, tt in dataloader:
            if np.isnan(tt.numpy()).mean() > 0.5:
                continue
            sequence[fillind, :] = ss
            target[fillind, :] = tt
            if fillind == sequence.shape[0] - 1:
                fillind = 0
            else:
                fillind += 1
                continue

            with torch.no_grad():
                encoding0 = net0(torch.Tensor(sequence.float()).transpose(1, 2).cuda())
            encoding1, encoding2, encoding4, encoding8, encoding16, encoding32 = net(encoding0)
            encodings = {
                1: encoding1,
                2: encoding2,
                4: encoding4,
                8: encoding8,
                16: encoding16,
                32: encoding32,
            }

            def train_step(level, start, coarse_pred=None):
                target_np = target[
                    :, start : start + 250 * level, start : start + 250 * level
                ].numpy()
                target_r = np.nanmean(
                    np.nanmean(
                        np.reshape(target_np, (target.shape[0], 250, level, 250, level)), axis=4
                    ),
                    axis=2,
                )
                distenc = torch.log(
                    torch.FloatTensor(normmats[level][None, None, :, :]).cuda()
                ).expand(sequence.shape[0], 1, 250, 250)
                if coarse_pred is not None:
                    pred = denets[level].forward(
                        encodings[level][:, :, int(start / level) : int(start / level) + 250],
                        distenc,
                        coarse_pred,
                    )
                else:
                    pred = denets[level].forward(
                        encodings[level][:, :, int(start / level) : int(start / level) + 250],
                        distenc,
                    )

                target_cuda = torch.Tensor(
                    np.log(((target_r + epss[level]) / (normmats[level] + epss[level])))[:, 0:, 0:]
                ).cuda()
                loss = (
                    (
                        pred[:, 0, 0:, 0:][~torch.isnan(target_cuda)]
                        - target_cuda[~torch.isnan(target_cuda)]
                    )
                    ** 2
                ).mean()
                loss = loss
                past_losses[level].append(loss.detach().cpu().numpy())
                return loss, pred

            start = 0
            loss32, pred = train_step(32, start)
            r = np.random.randint(0, 125)
            start = start + r * 32
            loss16, pred = train_step(16, start, pred[:, :, r : r + 125, r : r + 125].detach())
            r = np.random.randint(0, 125)
            start = start + r * 16
            loss8, pred = train_step(8, start, pred[:, :, r : r + 125, r : r + 125].detach())
            r = np.random.randint(0, 125)
            start = start + r * 8
            loss4, pred = train_step(4, start, pred[:, :, r : r + 125, r : r + 125].detach())
            r = np.random.randint(0, 125)
            start = start + r * 4
            loss2, pred = train_step(2, start, pred[:, :, r : r + 125, r : r + 125].detach())
            r = np.random.randint(0, 125)
            start = start + r * 2
            loss1, pred = train_step(1, start, pred[:, :, r : r + 125, r : r + 125].detach())
            loss = loss32 + loss16 + loss8 + loss4 + loss2 + loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del encodings
            del encoding1
            del encoding2
            del encoding4
            del encoding8
            del encoding16
            del encoding32
            del pred

            if i % 500 == 0:
                print("l1:" + str(np.mean(past_losses[1][-500:])), flush=True)
                print("l2:" + str(np.mean(past_losses[2][-500:])), flush=True)
                print("l4:" + str(np.mean(past_losses[4][-500:])), flush=True)
                print("l8:" + str(np.mean(past_losses[8][-500:])), flush=True)
                print("l16:" + str(np.mean(past_losses[16][-500:])), flush=True)
                print("l32:" + str(np.mean(past_losses[32][-500:])), flush=True)

            if i % 500 == 0:
                torch.save(net.state_dict(), "./models/model_" + modelstr + ".checkpoint")
                torch.save(denet_1.state_dict(), "./models/model_" + modelstr + ".d1.checkpoint")
                torch.save(denet_2.state_dict(), "./models/model_" + modelstr + ".d2.checkpoint")
                torch.save(denet_4.state_dict(), "./models/model_" + modelstr + ".d4.checkpoint")
                torch.save(denet_8.state_dict(), "./models/model_" + modelstr + ".d8.checkpoint")
                torch.save(denet_16.state_dict(), "./models/model_" + modelstr + ".d16.checkpoint")
                torch.save(denet_32.state_dict(), "./models/model_" + modelstr + ".d32.checkpoint")
                torch.save(optimizer.state_dict(), "./models/model_" + modelstr + ".optimizer")

            if i % 2000 == 0 and i != 0:
                t2 = Genomic2DFeatures(
                    ["../resources/4DNFILP99QJS.HCT_auxin6h.rebinned.mcool::/resolutions/4000"],
                    ["r1000"],
                    (8000, 8000),
                    cg=True,
                )
                t2.mode = "validate"
                sampler2 = RandomPositionsSamplerHiC(
                    reference_sequence=MemmapGenome(
                        input_path="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                        memmapfile="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
                        blacklist_regions="hg38",
                    ),
                    target=t2,
                    features=["r1000"],
                    test_holdout=["chr9", "chr10"],
                    validation_holdout=["chr8"],
                    sequence_length=32000000,
                    position_resolution=4000,
                    random_shift=0,
                    random_strand=True,
                    cross_chromosome=False,
                )

                sampler2.mode = "validate"
                dataloader2 = SamplerDataLoader(sampler2, num_workers=16, batch_size=1)

                net.eval()
                denet_1.eval()
                denet_2.eval()
                denet_4.eval()
                denet_8.eval()
                denet_16.eval()
                denet_32.eval()

                count = 0
                mses = {1: [], 2: [], 4: [], 8: [], 16: [], 32: []}
                corrs = {1: [], 2: [], 4: [], 8: [], 16: [], 32: []}
                fillind = 0
                for ss, tt in dataloader2:
                    if np.isnan(tt.numpy()).mean() > 0.3:
                        continue
                    sequence[fillind, :] = ss
                    target[fillind, :] = tt
                    if fillind == sequence.shape[0] - 1:
                        fillind = 0
                    else:
                        fillind += 1
                        continue

                    count += 1
                    if count == 100:
                        break
                    encoding1, encoding2, encoding4, encoding8, encoding16, encoding32 = net(
                        net0(torch.Tensor(sequence.float()).transpose(1, 2).cuda())
                    )

                    encodings = {
                        1: encoding1,
                        2: encoding2,
                        4: encoding4,
                        8: encoding8,
                        16: encoding16,
                        32: encoding32,
                    }

                    def eval_step(level, start, coarse_pred=None, plot=False):
                        target_r = np.nanmean(
                            np.nanmean(
                                np.reshape(
                                    target[
                                        :, start : start + 250 * level, start : start + 250 * level
                                    ].numpy(),
                                    (target.shape[0], 250, level, 250, level),
                                ),
                                axis=4,
                            ),
                            axis=2,
                        )
                        distenc = torch.log(
                            torch.FloatTensor(normmats[level][None, None, :, :]).cuda()
                        ).expand(sequence.shape[0], 1, 250, 250)
                        if coarse_pred is not None:
                            pred = denets[level].forward(
                                encodings[level][
                                    :, :, int(start / level) : int(start / level) + 250
                                ],
                                distenc,
                                coarse_pred,
                            )
                        else:
                            pred = denets[level].forward(
                                encodings[level][
                                    :, :, int(start / level) : int(start / level) + 250
                                ],
                                distenc,
                            )
                        target_cuda = torch.Tensor(
                            np.log(((target_r + epss[level]) / (normmats[level] + epss[level])))[
                                :, 0:, 0:
                            ]
                        ).cuda()
                        loss = (
                            (
                                pred[:, 0, 0:, 0:][~torch.isnan(target_cuda)]
                                - target_cuda[~torch.isnan(target_cuda)]
                            )
                            ** 2
                        ).mean()

                        mses[level].append(loss.detach().cpu().numpy())
                        pred_np = (
                            pred[:, 0, 0:, 0:].detach().cpu().numpy().reshape((pred.shape[0], -1))
                        )
                        target_np = np.log(
                            (target_r + epss[level]) / (normmats[level] + epss[level])
                        )[:, 0:, 0:].reshape((pred.shape[0], -1))

                        for j in range(pred_np.shape[0]):
                            validinds = ~np.isnan(target_np[j, :])
                            if np.mean(validinds) > 0.3:
                                corrs[level].append(
                                    pearsonr(pred_np[j, validinds], target_np[j, validinds])[0]
                                )
                            else:
                                corrs[level].append(np.nan)
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
                                    np.log(
                                        ((target_r + epss[level]) / (normmats[level] + epss[level]))
                                    )[ii, :, :],
                                    np=True,
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
                    pred = eval_step(32, start, plot=count == 99)
                    start = start + 63 * 32
                    pred = eval_step(16, start, pred[:, :, 63:188, 63:188], plot=count == 99)
                    start = start + 62 * 16
                    pred = eval_step(8, start, pred[:, :, 62:187, 62:187], plot=count == 99)
                    start = start + 62 * 8
                    pred = eval_step(4, start, pred[:, :, 62:187, 62:187], plot=count == 99)
                    start = start + 62 * 4
                    pred = eval_step(2, start, pred[:, :, 62:187, 62:187], plot=count == 99)
                    start = start + 62 * 2
                    pred = eval_step(1, start, pred[:, :, 62:187, 62:187], plot=count == 99)

                    del encodings
                    del encoding1
                    del encoding2
                    del encoding4
                    del encoding8
                    del encoding16
                    del encoding32
                    del pred

                print(
                    "Cor1 {0}, Cor2 {1}, Cor4 {2}, Cor8 {3}, Cor16 {4}, Cor32 {5}".format(
                        np.nanmean(corrs[1]),
                        np.nanmean(corrs[2]),
                        np.nanmean(corrs[4]),
                        np.nanmean(corrs[8]),
                        np.nanmean(corrs[16]),
                        np.nanmean(corrs[32]),
                    )
                )
                print(
                    "MSE1 {0}, MSE2 {1}, MSE4 {2}, MSE8 {3}, MSE16 {4}, MSE32 {5}".format(
                        np.nanmean(mses[1]),
                        np.nanmean(mses[2]),
                        np.nanmean(mses[4]),
                        np.nanmean(mses[8]),
                        np.nanmean(mses[16]),
                        np.nanmean(mses[32]),
                    )
                )
                net.train()
                denet_1.train()
                denet_2.train()
                denet_4.train()
                denet_8.train()
                denet_16.train()
                denet_32.train()
            i += 1

