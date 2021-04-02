"""
This script process the compute-expected output from cooltools, 
combine expectations across chromosomes, apply smoothing 
(note the smoothing settings may need to be adjusted
for your data) and transform to format the Orca uses. It outputs
three files, the cis-interaction expectation .npy file, the
monotonically transformed cis-interaction expectation .mono.npy file 
(only differs from the non-monotonically transformed in very high distances),
and a trans-interaction expectation .trans.npy file (contains only a scalar).


Example usage: python process_expectation.py expectation_file_res1000 1000

"""
import pandas as pd
import sys
import numpy as np

res = int(sys.argv[2])

expected = pd.read_csv(sys.argv[1], sep="\t")
expectedsum = expected.groupby(["diag"]).agg({"n_valid": "sum", "balanced.sum": "sum"})
expectedsum["balanced.avg"] = expectedsum["balanced.sum"] / expectedsum["n_valid"]

from statsmodels.nonparametric.smoothers_lowess import lowess

v = np.log(expectedsum["balanced.avg"].values)
v = v[: np.min(np.argwhere(~np.isfinite(v)))]
sv0 = lowess(
    v[int(400 / (res / 4000)) :], np.log(np.arange(int(400 / (res / 4000)), len(v)) + 1), frac=0.01
)[:, 1]
sv2 = lowess(
    v[int(400 / (res / 4000)) :], np.log(np.arange(int(400 / (res / 4000)), len(v)) + 1), frac=0.1
)[:, 1]
sv = np.hstack(
    [
        v[: int(400 / (res / 4000))],
        sv0[: int(10000 / (res / 4000))],
        sv2[int(10000 / (res / 4000)) :],
    ]
)

sv_mono = np.minimum.accumulate(sv)

np.save(arr=sv, file=sys.argv[1] + ".npy")
np.save(arr=sv_mono, file=sys.argv[1] + ".mono.npy")

expectedtrans = pd.read_csv(sys.argv[1] + ".trans", sep="\t")

np.save(
    arr=np.log(np.sum(expectedtrans["balanced.sum"]) / np.sum(expectedtrans["n_valid"])),
    file=sys.argv[1] + ".trans.npy",
)

