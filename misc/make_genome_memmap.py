"""
This script converts the genome fasta file to a memory map
file of genome one-hot encodings. This is used to accelerate
genome sequence retrieval (Orca automatically detects the memmap
file and use it) and is required if you use the training scripts.

Example usage: python make_genome_memmap.py
"""

import sys
import pathlib

ORCA_PATH = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.append(ORCA_PATH)
from selene_utils2 import MemmapGenome

if __name__ == "__main__":
    hg38 = MemmapGenome(
        input_path=ORCA_PATH + "/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
        memmapfile=ORCA_PATH + "/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
        init_unpicklable=True,
    )
    print(hg38.initialized,flush=True)
