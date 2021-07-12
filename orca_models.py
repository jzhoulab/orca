"""
This module contains the class definition of all Orca models.
For usage of the models, see the orca_predict module.
"""
import pathlib
import numpy as np

import torch
from torch import nn

from orca_modules import Encoder, Encoder2, Encoder3, Decoder, Decoder_1m, Net

ORCA_PATH = str(pathlib.Path(__file__).parent.absolute())


class H1esc(nn.Module):
    """
    Orca H1-ESC model (1-32Mb) 

    Attributes
    ----------
    net0 : nn.DataParallel(Encoder)
        The first section of the multi-resolution encoder 
        (bp resolution to 4kb resolution).
    net : nn.DataParallel(Encoder2)
        The second section of the multi-resolution encoder 
        (4kb resolution to 128kb resolution).
    denets : dict(int: nn.DataParallel(Decoder))
        Decoders at each level, which are stored in a dictionary
        with an integer as key.
    normmats : dict(int: numpy.ndarray)
        The distance-based background matrices with expected log
        fold over background values at each level.
    epss : dict(int: float)
        The minimum background value at each level. Used for 
        stablizing the log fold computation by adding
        to both the nominator and the denominator.
    """

    def __init__(self,):
        super(H1esc, self).__init__()
        modelstr = "h1esc"
        self.net = nn.DataParallel(Encoder2())
        self.denet_1 = nn.DataParallel(Decoder())
        self.denet_2 = nn.DataParallel(Decoder())
        self.denet_4 = nn.DataParallel(Decoder())
        self.denet_8 = nn.DataParallel(Decoder())
        self.denet_16 = nn.DataParallel(Decoder())
        self.denet_32 = nn.DataParallel(Decoder())

        num_threads = torch.get_num_threads()
        self.net.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".net.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )
        self.denet_1.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d1.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )
        self.denet_2.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d2.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )
        self.denet_4.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d4.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )
        self.denet_8.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d8.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )
        self.denet_16.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d16.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )
        self.denet_32.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d32.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )

        self.net0 = nn.DataParallel(Encoder())
        net0_dict = self.net0.state_dict()

        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_" + modelstr + ".net0.statedict",
            map_location=torch.device("cpu"),
        )

        pretrained_dict_filtered = {key: pretrained_dict["module." + key] for key in net0_dict}
        self.net0.load_state_dict(pretrained_dict_filtered)

        self.denet_1_pt = nn.DataParallel(Decoder_1m())
        denet_1_pt_dict = self.denet_1_pt.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_" + modelstr + ".net0.statedict",
            map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {
            key: pretrained_dict["module." + key] for key in denet_1_pt_dict
        }
        self.denet_1_pt.load_state_dict(pretrained_dict_filtered)

        self.denet_1_pt.eval()
        self.net0.eval()
        self.net.eval()
        self.denet_1.eval()
        self.denet_2.eval()
        self.denet_4.eval()
        self.denet_8.eval()
        self.denet_16.eval()
        self.denet_32.eval()

        expected_log = np.load(
            ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res4000.npy"
        )

        normmat = np.exp(expected_log[np.abs(np.arange(8000)[None, :] - np.arange(8000)[:, None])])

        normmat_r1 = np.reshape(normmat[:250, :250], (250, 1, 250, 1)).mean(axis=1).mean(axis=2)
        normmat_r2 = np.reshape(normmat[:500, :500], (250, 2, 250, 2)).mean(axis=1).mean(axis=2)
        normmat_r4 = np.reshape(normmat[:1000, :1000], (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        normmat_r8 = np.reshape(normmat[:2000, :2000], (250, 8, 250, 8)).mean(axis=1).mean(axis=2)
        normmat_r16 = (
            np.reshape(normmat[:4000, :4000], (250, 16, 250, 16)).mean(axis=1).mean(axis=2)
        )
        normmat_r32 = (
            np.reshape(normmat[:8000, :8000], (250, 32, 250, 32)).mean(axis=1).mean(axis=2)
        )
        eps1 = np.min(normmat_r1)
        eps2 = np.min(normmat_r2)
        eps4 = np.min(normmat_r4)
        eps8 = np.min(normmat_r8)
        eps16 = np.min(normmat_r16)
        eps32 = np.min(normmat_r32)

        self.normmats = {
            1: normmat_r1,
            2: normmat_r2,
            4: normmat_r4,
            8: normmat_r8,
            16: normmat_r16,
            32: normmat_r32,
        }
        self.epss = {1: eps1, 2: eps2, 4: eps4, 8: eps8, 16: eps16, 32: eps32}
        self.denets = {
            1: self.denet_1,
            2: self.denet_2,
            4: self.denet_4,
            8: self.denet_8,
            16: self.denet_16,
            32: self.denet_32,
        }
        torch.set_num_threads(num_threads)


class Hff(nn.Module):
    """
    Orca HFF model (1-32Mb) 

    Attributes
    ----------
    net0 : nn.DataParallel(Encoder)
        The first section of the multi-resolution encoder 
        (bp resolution to 4kb resolution).
    net : nn.DataParallel(Encoder2)
        The second section of the multi-resolution encoder 
        (4kb resolution to 128kb resolution).
    denets : dict(int: nn.DataParallel(Decoder))
        Decoders at each level, which are stored in a dictionary
        with an integer as key.
    normmats : dict(int: numpy.ndarray)
        The distance-based background matrices with expected log
        fold over background values at each level.
    epss : dict(int: float)
        The minimum background value at each level. Used for 
        stablizing the log fold computation by adding
        to both the nominator and the denominator.
    """

    def __init__(self):
        super(Hff, self).__init__()
        modelstr = "hff"
        self.net = nn.DataParallel(Encoder2())
        self.denet_1 = nn.DataParallel(Decoder())
        self.denet_2 = nn.DataParallel(Decoder())
        self.denet_4 = nn.DataParallel(Decoder())
        self.denet_8 = nn.DataParallel(Decoder())
        self.denet_16 = nn.DataParallel(Decoder())
        self.denet_32 = nn.DataParallel(Decoder())

        num_threads = torch.get_num_threads()
        self.net.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".net.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )
        self.denet_1.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d1.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )
        self.denet_2.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d2.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )
        self.denet_4.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d4.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )
        self.denet_8.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d8.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )
        self.denet_16.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d16.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )
        self.denet_32.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d32.statedict",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )

        self.net0 = nn.DataParallel(Encoder())
        net0_dict = self.net0.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_" + modelstr + ".net0.statedict",
            map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {key: pretrained_dict["module." + key] for key in net0_dict}
        self.net0.load_state_dict(pretrained_dict_filtered)

        self.denet_1_pt = nn.DataParallel(Decoder_1m())
        denet_1_pt_dict = self.denet_1_pt.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_" + modelstr + ".net0.statedict",
            map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {
            key: pretrained_dict["module." + key] for key in denet_1_pt_dict
        }
        self.denet_1_pt.load_state_dict(pretrained_dict_filtered)

        self.denet_1_pt.eval()
        self.net0.eval()
        self.net.eval()
        self.denet_1.eval()
        self.denet_2.eval()
        self.denet_4.eval()
        self.denet_8.eval()
        self.denet_16.eval()
        self.denet_32.eval()

        expected_log = np.load(
            ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool.expected.res4000.npy"
        )
        normmat = np.exp(expected_log[np.abs(np.arange(8000)[:, None] - np.arange(8000)[None, :])])

        normmat_r1 = np.reshape(normmat[:250, :250], (250, 1, 250, 1)).mean(axis=1).mean(axis=2)
        normmat_r2 = np.reshape(normmat[:500, :500], (250, 2, 250, 2)).mean(axis=1).mean(axis=2)
        normmat_r4 = np.reshape(normmat[:1000, :1000], (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        normmat_r8 = np.reshape(normmat[:2000, :2000], (250, 8, 250, 8)).mean(axis=1).mean(axis=2)
        normmat_r16 = (
            np.reshape(normmat[:4000, :4000], (250, 16, 250, 16)).mean(axis=1).mean(axis=2)
        )
        normmat_r32 = (
            np.reshape(normmat[:8000, :8000], (250, 32, 250, 32)).mean(axis=1).mean(axis=2)
        )
        eps1 = np.min(normmat_r1)
        eps2 = np.min(normmat_r2)
        eps4 = np.min(normmat_r4)
        eps8 = np.min(normmat_r8)
        eps16 = np.min(normmat_r16)
        eps32 = np.min(normmat_r32)

        self.normmats = {
            1: normmat_r1,
            2: normmat_r2,
            4: normmat_r4,
            8: normmat_r8,
            16: normmat_r16,
            32: normmat_r32,
        }
        self.epss = {1: eps1, 2: eps2, 4: eps4, 8: eps8, 16: eps16, 32: eps32}
        self.denets = {
            1: self.denet_1,
            2: self.denet_2,
            4: self.denet_4,
            8: self.denet_8,
            16: self.denet_16,
            32: self.denet_32,
        }
        torch.set_num_threads(num_threads)


class HCTnoc(nn.Module):
    """
    Orca cohesin-depleted HCT116 model (1-32Mb) 

    Attributes
    ----------
    net0 : nn.DataParallel(Encoder)
        The first section of the multi-resolution encoder 
        (bp resolution to 4kb resolution).
    net : nn.DataParallel(Encoder2)
        The second section of the multi-resolution encoder 
        (4kb resolution to 128kb resolution).
    denets : dict(int: nn.DataParallel(Decoder))
        Decoders at each level, which are stored in a dictionary
        with an integer as key.
    normmats : dict(int: numpy.ndarray)
        The distance-based background matrices with expected log
        fold over background values at each level.
    epss : dict(int: float)
        The minimum background value at each level. Used for 
        stablizing the log fold computation by adding
        to both the nominator and the denominator.
    """

    def __init__(self):
        super(HCTnoc, self).__init__()
        modelstr = "hctnoc"

        self.net = nn.DataParallel(Encoder2())
        self.denet_1 = nn.DataParallel(Decoder())
        self.denet_2 = nn.DataParallel(Decoder())
        self.denet_4 = nn.DataParallel(Decoder())
        self.denet_8 = nn.DataParallel(Decoder())
        self.denet_16 = nn.DataParallel(Decoder())
        self.denet_32 = nn.DataParallel(Decoder())

        self.net.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".net.statedict"), strict=True
        )
        self.denet_1.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".d1.statedict"), strict=True
        )
        self.denet_2.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".d2.statedict"), strict=True
        )
        self.denet_4.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".d4.statedict"), strict=True
        )
        self.denet_8.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".d8.statedict"), strict=True
        )
        self.denet_16.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".d16.statedict"), strict=True
        )
        self.denet_32.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".d32.statedict"), strict=True
        )

        self.net0 = nn.DataParallel(Encoder())
        self.net0.load_state_dict(
            torch.load(ORCA_PATH + "/models/orca_" + modelstr + ".net0.statedict"), strict=True
        )
        self.net0.cuda()
        self.net0.eval()
        self.net.eval()
        self.denet_1.eval()
        self.denet_2.eval()
        self.denet_4.eval()
        self.denet_8.eval()
        self.denet_16.eval()
        self.denet_32.eval()

        smooth_diag = np.load(
            ORCA_PATH + "/resources/4DNFILP99QJS.HCT_auxin6h.rebinned.mcool.expected.res4000.npy"
        )

        normmat = np.exp(smooth_diag[np.abs(np.arange(8000)[None, :] - np.arange(8000)[:, None])])

        normmat_r1 = np.reshape(normmat[:250, :250], (250, 1, 250, 1)).mean(axis=1).mean(axis=2)
        normmat_r2 = np.reshape(normmat[:500, :500], (250, 2, 250, 2)).mean(axis=1).mean(axis=2)
        normmat_r4 = np.reshape(normmat[:1000, :1000], (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        normmat_r8 = np.reshape(normmat[:2000, :2000], (250, 8, 250, 8)).mean(axis=1).mean(axis=2)
        normmat_r16 = (
            np.reshape(normmat[:4000, :4000], (250, 16, 250, 16)).mean(axis=1).mean(axis=2)
        )
        normmat_r32 = (
            np.reshape(normmat[:8000, :8000], (250, 32, 250, 32)).mean(axis=1).mean(axis=2)
        )
        eps1 = np.min(normmat_r1)
        eps2 = np.min(normmat_r2)
        eps4 = np.min(normmat_r4)
        eps8 = np.min(normmat_r8)
        eps16 = np.min(normmat_r16)
        eps32 = np.min(normmat_r32)

        self.normmats = {
            1: normmat_r1,
            2: normmat_r2,
            4: normmat_r4,
            8: normmat_r8,
            16: normmat_r16,
            32: normmat_r32,
        }
        self.epss = {1: eps1, 2: eps2, 4: eps4, 8: eps8, 16: eps16, 32: eps32}
        self.denets = {
            1: self.denet_1,
            2: self.denet_2,
            4: self.denet_4,
            8: self.denet_8,
            16: self.denet_16,
            32: self.denet_32,
        }


class H1esc_1M(nn.Module):
    """
    Orca H1-ESC model (1Mb) 

    Attributes
    ----------
    net : nn.DataParallel(Net)
        Integrated Encoder and Decoder for 1Mb model.
    normmats : dict(int: numpy.ndarray)
        The distance-based background matrices with expected log
        fold over background values at each level.
    epss : dict(int: float)
        The minimum background value at each level. Used for 
        stablizing the log fold computation by adding
        to both the nominator and the denominator.
    """

    def __init__(self,):
        super(H1esc_1M, self).__init__()
        self.net = nn.DataParallel(Net(num_1d=32))
        num_threads = torch.get_num_threads()
        net_dict = self.net.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_h1esc.net0.statedict", map_location=torch.device("cpu")
        )
        pretrained_dict_filtered = {key: pretrained_dict["module." + key] for key in net_dict}
        self.net.load_state_dict(pretrained_dict_filtered)
        self.net.eval()

        expected_log = np.load(
            ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res1000.npy"
        )[:1000]

        normmat = np.exp(expected_log[np.abs(np.arange(1000)[None, :] - np.arange(1000)[:, None])])

        normmat_r = np.reshape(normmat, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        eps = np.min(normmat_r)

        self.normmats = {1: normmat_r}
        self.epss = {1: eps}
        torch.set_num_threads(num_threads)

    def forward(self, x):
        pred, _ = self.net.forward(x)

        return pred


class Hff_1M(nn.Module):
    """
    Orca HFF model (1Mb) 

    Attributes
    ----------
    net : nn.DataParallel(Net)
        Integrated Encoder and Decoder for 1Mb model.
    normmats : dict(int: numpy.ndarray)
        The distance-based background matrices with expected log
        fold over background values at each level.
    epss : dict(int: float)
        The minimum background value at each level. Used for 
        stablizing the log fold computation by adding
        to both the nominator and the denominator.
    """

    def __init__(self,):
        super(Hff_1M, self).__init__()
        self.net = nn.DataParallel(Net(num_1d=22))
        num_threads = torch.get_num_threads()
        net_dict = self.net.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_hff.net0.statedict", map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {key: pretrained_dict["module." + key] for key in net_dict}
        self.net.load_state_dict(pretrained_dict_filtered)
        self.net.eval()

        expected = np.exp(
            np.load(ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool.expected.res1000.npy")[
                :1000
            ]
        )
        normmat = expected[np.abs(np.arange(1000)[:, None] - np.arange(1000)[None, :])]

        normmat_r = np.reshape(normmat, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        eps = np.min(normmat_r)

        self.normmats = {1: normmat_r}
        self.epss = {1: eps}
        torch.set_num_threads(num_threads)

    def forward(self, x):
        pred, _ = self.net.forward(x)
        return pred


class H1esc_256M(nn.Module):
    """
    Orca H1-ESC model (32-256Mb) 

    Attributes
    ----------
    net0 : nn.DataParallel(Encoder)
        The first section of the multi-resolution encoder 
        (bp resolution to 4kb resolution).
    net1 : nn.DataParallel(Encoder2)
        The second section of the multi-resolution encoder 
        (4kb resolution to 128kb resolution).
    net : nn.DataParallel(Encoder3)
        The third section of the multi-resolution encoder 
        (128kb resolution to 1024kb resolution).
    denets : dict(int: nn.DataParallel(Decoder))
        Decoders at each level, which are stored in a dictionary
        with an integer as key.
    normmats : dict(int: numpy.ndarray)
        The distance-based background matrices with expected log
        fold over background values at each level.
    epss : dict(int: float)
        The minimum background value at each level. Used for 
        stablizing the log fold computation by adding
        to both the nominator and the denominator.
    """

    def __init__(self,):
        super(H1esc_256M, self).__init__()
        modelstr = "h1esc_256m"
        self.net = nn.DataParallel(Encoder3())
        self.denet_32 = nn.DataParallel(Decoder())
        self.denet_64 = nn.DataParallel(Decoder())
        self.denet_128 = nn.DataParallel(Decoder())
        self.denet_256 = nn.DataParallel(Decoder())

        num_threads = torch.get_num_threads()
        self.net.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".net.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_32.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d32.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_64.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d64.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_128.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d128.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_256.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d256.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.net0 = nn.DataParallel(Encoder())
        net0_dict = self.net0.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_h1esc.net0.statedict", map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {key: pretrained_dict["module." + key] for key in net0_dict}
        self.net0.load_state_dict(pretrained_dict_filtered)

        self.net1 = nn.DataParallel(Encoder2())
        net1_dict = self.net1.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_h1esc.net.statedict", map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {key: pretrained_dict[key] for key in net1_dict}
        self.net1.load_state_dict(pretrained_dict_filtered)
        self.net0.eval()
        self.net1.eval()
        self.net.eval()
        self.denet_32.eval()
        self.denet_64.eval()
        self.denet_128.eval()
        self.denet_256.eval()
        self.background_cis = np.load(
            ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res32000.mono.npy"
        )
        self.background_trans = np.load(
            ORCA_PATH + "/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res32000.trans.npy"
        )
        self.background_cis = np.hstack([np.exp(self.background_cis), np.repeat(np.nan, 2000)])
        self.background_trans = np.exp(self.background_trans)

        self.denets = {
            32: self.denet_32,
            64: self.denet_64,
            128: self.denet_128,
            256: self.denet_256,
        }
        torch.set_num_threads(num_threads)


class Hff_256M(nn.Module):
    """
    Orca HFF model (32-256Mb) 

    Attributes
    ----------
    net0 : nn.DataParallel(Encoder)
        The first section of the multi-resolution encoder 
        (bp resolution to 4kb resolution).
    net1 : nn.DataParallel(Encoder2)
        The second section of the multi-resolution encoder 
        (4kb resolution to 128kb resolution).
    net : nn.DataParallel(Encoder3)
        The third section of the multi-resolution encoder 
        (128kb resolution to 1024kb resolution).
    denets : dict(int: nn.DataParallel(Decoder))
        Decoders at each level, which are stored in a dictionary
        with an integer as key.
    normmats : dict(int: numpy.ndarray)
        The distance-based background matrices with expected log
        fold over background values at each level.
    epss : dict(int: float)
        The minimum background value at each level. Used for 
        stablizing the log fold computation by adding
        to both the nominator and the denominator.
    """

    def __init__(self):
        super(Hff_256M, self).__init__()
        modelstr = "hff_256m"

        self.net = nn.DataParallel(Encoder3())
        self.denet_32 = nn.DataParallel(Decoder())
        self.denet_64 = nn.DataParallel(Decoder())
        self.denet_128 = nn.DataParallel(Decoder())
        self.denet_256 = nn.DataParallel(Decoder())

        num_threads = torch.get_num_threads()
        self.net.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".net.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_32.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d32.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_64.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d64.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_128.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d128.statedict",
                map_location=torch.device("cpu"),
            )
        )
        self.denet_256.load_state_dict(
            torch.load(
                ORCA_PATH + "/models/orca_" + modelstr + ".d256.statedict",
                map_location=torch.device("cpu"),
            )
        )

        self.net0 = nn.DataParallel(Encoder())
        net0_dict = self.net0.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_hff.net0.statedict", map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {key: pretrained_dict["module." + key] for key in net0_dict}
        self.net0.load_state_dict(pretrained_dict_filtered)

        self.net1 = nn.DataParallel(Encoder2())
        net1_dict = self.net1.state_dict()
        pretrained_dict = torch.load(
            ORCA_PATH + "/models/orca_hff.net.statedict", map_location=torch.device("cpu"),
        )
        pretrained_dict_filtered = {key: pretrained_dict[key] for key in net1_dict}
        self.net1.load_state_dict(pretrained_dict_filtered)

        self.net0.eval()
        self.net1.eval()
        self.net.eval()
        self.denet_32.eval()
        self.denet_64.eval()
        self.denet_128.eval()
        self.denet_256.eval()

        self.background_cis = np.load(
            ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool.expected.res32000.mono.npy"
        )
        self.background_trans = np.load(
            ORCA_PATH + "/resources/4DNFI643OYP9.rebinned.mcool.expected.res32000.trans.npy"
        )
        self.background_cis = np.hstack([np.exp(self.background_cis), np.repeat(np.nan, 2000)])
        self.background_trans = np.exp(self.background_trans)

        self.denets = {
            32: self.denet_32,
            64: self.denet_64,
            128: self.denet_128,
            256: self.denet_256,
        }
        torch.set_num_threads(num_threads)
