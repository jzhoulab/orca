
<p align="center">
  <img  height="140" src="https://github.com/jzhoulab/orca/blob/main/docs/source/orca_logo.png">
</p>

# Orca

This repository contains code for Orca, a deep learning sequence modeling framework for multiscale genome structure prediction. Orca can **predict genome interactions from kilobase to whole-chromosome-scales** using only genomic sequence as input.  The manuscript is now published [here](https://www.nature.com/articles/s41588-022-01065-4).

This is our main repository for Orca, including code for applying Orca models or training new models. For reproducing the analyses in our manuscript,  please visit our manuscript [repository](https://github.com/jzhoulab/orca_manuscript).  The full API documentation is available [here](http://jzhoulab.github.io/orca-docs/). A GPU-backed webserver for running the core functionalities of Orca is also available at: [orca.zhoulab.io](https://orca.zhoulab.io). 

### What can I use Orca for?

-  Predict the genome structural impacts of any genome variant, including large structural variants of almost any size (the maximum input size 256Mb, which is larger than the largest human chromosome).  
-  Predict the genome 3D structure from any human genome sequence,  which means you can introduce multiple variants,  haplotypes, an entire assembled genome,  or any sequence you wish to perform an in silico Hi-C experiment on. 
-   Analyze sequence dependencies of genome 3D structure by performing virtual genetic screens.  Orca sequence models can serve as an “in silico genome observatory” that allows designing and performing virtual genetic screens to probe the sequence basis of genome 3D organization.


###  What is Orca?

Orca is a deep learning sequence modeling framework for multiscale genome interaction prediction. Orca models are trained on high-resolution micro-C datasets for H1-ESC and HFF cell lines (and a cohesin-depleted HCT116 Hi-C model for the analysis of sequence dependencies of chromatin compartments). If you have sufficient computational resources including GPUs,  you can also train your own models on Hi-C type data given any cooler format input following our examples (see the training section).
<p align="center">
  <img  src="https://github.com/jzhoulab/orca/blob/main/docs/source/orca_diagram.png">
</p>

### Get started 
If you just need predictions for one or a handful of variants, we have provided the core functionalities on a web server: [orca.zhoulab.io](https://orca.zhoulab.io). 

### Install and run Orca locally

#### Pre-requisites: Python environment and Selene library.

1. orca_env Python environment

    The enviornment setup involves 3 steps, install (1) Python 3.9, (2) Pytorch, and (3) the remaining packages.
    This is to save time for conda to solve the dependency for Pytorch.

  - Install [conda/miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don't have it already.
  - Create the environment from the orca_env_part1.yml file: `conda env create -f orca_env_part1.yml`
  - Activate the environment: `conda activate orca_env`
  - Install Pytorch following the [Pytorch installation guide](https://pytorch.org/get-started/locally/), choose the appropriate parameters (CPU or GPU, OS, etc.) for your system.
  - Install the remaining packages: `conda env update -f orca_env_part2.yml`

2. Install Selene (under the orca_env environment)
  ```bash
  git clone https://github.com/kathyxchen/selene.git
  cd selene
  git checkout custom_target_support
  python setup.py build_ext --inplace
  python setup.py install 
  ```

Now you are ready to run Orca locally, with the orca repository cloned and resource files downloaded.

#### Clone the Orca repository
```bash
git clone https://github.com/jzhoulab/orca.git
cd orca
```
#### Download model and relevant resource files
Next, download the model and other resource files needed by Orca. Because these files are large and some are optional, we packaged into several files and you can download what you need (some functionalities may not be available if the relevant files are not downloaded).  

The minimal resource package for running Orca are packaged [here](https://zenodo.org/record/6234936/files/resources_core.tar.gz) (1.3G),  which includes the Orca models and the  hg38 reference genome. It is also recommended to download the preprocessed micro-C datasets binned to the resolutions that Orca use,  which will allow for comparisons with observed data, from [here](https://zenodo.org/record/6234936/files/resources_mcools.tar.gz) (34G). In addition, if you would like to generate chromatin tracks visualizations,  you can download these files [here](https://zenodo.org/record/4594676/files/resources_extra.tar.gz) (15G). To download and extract all resource files, you can run the following commands under the orca directory.

```bash
wget https://zenodo.org/record/6234936/files/resources_core.tar.gz
wget https://zenodo.org/record/6234936/files/resources_mcools.tar.gz
wget https://zenodo.org/record/4594676/files/resources_extra.tar.gz
tar xf resources_core.tar.gz
tar xf resources_mcools.tar.gz
tar xf resources_extra.tar.gz
```

#### Basic Usage

You can use Orca through either the command-line-interface (CLI) which supports most of the core functionalities or access the full capabilities through python. You can jump to the [CLI](#orca-command-line-interface-cli) if you wish to just use the CLI.  

For using Orca from python, you can just add the directory you cloned to your python PATH `sys.path.append(ORCA_PATH)` and use it (for now we haven't made Orca a python package, because most of its functionalities depends on the resource files which is rather large).  The full API documentation is available [here](http://jzhoulab.github.io/orca-docs/).

To use Orca for any prediciton tasks, the first step is to load the models. The following scripts load the Orca models (32Mb and 256Mb).  

```python
import orca_predict
#Default is using GPU. Set use_cuda = False in load_resources to use CPU.
orca_predict.load_resources(models=['32M', '256M'])
```

For predicting multiscale 3D genome effects of simple structural variants, here are some examples:

```python
from orca_predict import *

#duplication variant
outputs = process_dup('chr17', 70845859, 71884859, hg38, file='./chr17_70845859_71884859_RevSex_dup_maximum_Sox9', show_genes=True)

#deletion variant
outputs = process_del('chr2', 220295000, 222000000, hg38, file='./chr2_220295000_222000000_del_Brachydactyly_Pax3', show_genes=True)

#inversion variant
outputs = process_inv('chr2', 218875000, 220155000, hg38, file='./chr2_218875000_220155000_inv_FsyndromeF1_Wnt6', show_genes=True)

#insertion variant, where ins_seq is the inserted sequence string
outputs = process_ins('chr6', 74353593, ins_seq, hg38, strand='+', file='./chr6_74353593_ins', show_genes=True)

#translocation variant, see docs for how such variant is expressed
outputs = process_single_breakpoint('chr5', 89411065, 'chr7', 94378248,'+','-', hg38, target=True, file='./chr5_89411065_chr7_94378248_+-', show_genes=True)

#Default is using the 1-32Mb models. To use the 256Mb models, just add window_radius=128000000
outputs = process_dup('chr17', 70845859, 71884859, hg38, window_radius=128000000, file='./chr17_70845859_71884859_RevSex_dup_maximum_Sox9')

#of course you can also predict any genomic region
outputs = process_region('chr9', 110404000, 111404000, hg38, window_radius=16000000, file='./chr17_110404000_111404000')
```
####  Usage - complex variants
You can also specify more complex variants than the above types, using the custom variant function (here we specify a simple two-segment variant, but it can be used to specify complex variants with an arbitrary number of segments)
```python
_ = process_custom([['chr5', 89411065, 89411065+16000000, '-'],
                ['chr7', 94378248, 94378248+16000000,'+']],
              [['chr5', 89411065-16000000, 89411065+16000000],
               ['chr7', 94378248-16000000, 94378248+16000000]],
              16000000, hg38, anno_list=[[16000000,'double']],
              ref_anno_list=[[16000000,32000000,'black'],[16000000,32000000,'black']], target=False, file='./custom_variant')
``` 

For even more flexibility,  you can just generate the sequence and give it to Orca to predict. This gives you the full control for introducing any variant, haplotype, or even an individual genome,  and predict the multiscale genome organization. 

```python
from selene_sdk.sequences import Genome
#sequence: 32Mb sequence
#mpos: specifies the coordinate to zoom into for multiscale prediction
#wpos: specifies the coordinate of the center position of the sequence. 
#chrom: chromosome name
outputs = genomepredict(Genome.sequence_to_encoding(sequence)[None,:,:], chrom, mpos=pos, wpos=wpos, use_cuda=True)

#For 256Mb sequence. See the docs for details of the input 
outputs = genomepredict_256Mb(Genome.sequence_to_encoding(sequence)[None,:,:], chrom, normmats, chrlen, mpos=pos, wpos=wpos, use_cuda=True)
```

For full information about using Orca,  you may visit our API documentation page (http://jzhoulab.github.io/orca-docs/).

#### Orca Command-line Interface (CLI)
For prediction of multiscale interactions for genomic regions,  structural variant of deletion, duplication, inversion, and translocation with single junctions,  you can use the command line interface orca_predict.py and the output includes graphical visualizations in pdf format and numerical results saved in pytorch serialization format. 

```docs
    Orca multiscale genome interaction sequence model prediction tool.

    Usage:
    orca_predict region [options] <coordinate> <output_dir>
    orca_predict del [options] <coordinate> <output_dir>
    orca_predict dup [options] <coordinate> <output_dir>
    orca_predict inv [options] <coordinate> <output_dir>
    orca_predict break [options] <coordinate> <output_dir>

    Options:
    -h --help       Show this screen.
    --show_genes    Show gene annotation (only supported for 32Mb models).
    --show_tracks   Show chromatin tracks (only supported for 32Mb models).
    --256m          Use 256Mb models (default is 32Mb).
    --nocuda        Use CPU implementation. 
    --version       Show version.
```

For input,  a `<coordinate>`  argument and an output directory `<output_dir>` are required.  Most prediction modes except for `break` requires specifying a region as input in the format like `chr9:94904000-126904000`.  

The `break` mode is used for predicting the effect of an translocation that connects two chromosomal breakpoints.  An example `<coordinate>`  input is `chr1:85691449 chr5:89533745 +/+`. Specifically, two breakpoint positions and the corresponding two orientations are needed. The orientations decide how the breakpoints are connected. The ‘+’ or ‘-’ sign indicate whether the left or right side of the breakpoint is used (not forward and reverse strands). For example ‘+/+’ indicates connecting chr1:0-85691449 with chr5:0-89533745 at the breakpoints, while ‘+/-’ indicates connecting chr1:0-85691449 with chr5:89533745-chromosome end. 

If the function that you needed is not available from the CLI,  you can use the python interface as introduced in the previous section and detailed in the Orca API documentation.



### Example output

As an example output, here we showed visualizations generated for the predictions of a duplication variant.  For structural variant prediction, Orca generates multiple files that each contains a series of multi-level predictions zooming into a breakpoint of the variant, or the corresponding position(s) of the breakpoint in the reference sequence.

The positions of the variant boundaries are be annotated on the left side of each plot. For reference sequence, we also show the micro-C data for comparison.

Example reference sequence predictions for duplication variant (breakpoint): 
<p align="center">
  <img  src="https://github.com/jzhoulab/orca/blob/main/docs/source/sv_pred_revsexdup_ref_r.png">
</p>


Example alternative sequence predictions for duplication variant (right boundary): 
<p align="center">
  <img  src="https://github.com/jzhoulab/orca/blob/main/docs/source/sv_pred_revsexdup_alt.png">
</p>

If you ask Orca to generate gene annotations or chromatin tracks,  you will also find annotation pdf outputs files which correspond to the same multi-level regions the genome interaction predictions are made on.

Besides graphical output, we also save the numerical predictions in PyTorch serialization format (pickle) with extension '.pth'. The .pth file can be loaded with `torch.load` or other pickle file loading function. Each file contains a python dictionary. If the prediction mode is one of the structural variant prediction modes, the dictionary stores multiple dictionaries each corresponding to an output file as described above. The dictionary includes:

`predictions` - Multi-level predictions for H1-ESC and HFF cell types.

`experiments` - Observations for H1-ESC and HFF cell types that matches the predictions (only available for reference allele).

`chr` - The chromosome name

`start_coords` - Start coordinates for the prediction at each level.

`end_coords` - End coordinates for the prediction at each level.

`annos` - Annotation information. A list indicating the relative variant positions for each interaction matrix, saved for plotting purpose.

### Orca-1Mb model

If you are running a larger scale virtural screen that focus mostly on  <1Mb structures, Orca-1Mb model offers better speed. Orca-1Mb can be used with the following code

```
import orca_predict
orca_predict.load_resources(models=["1M"])
from orca_predict import *
sequence = hg38.get_encoding_from_coords(chrm, start, start + 1000000)[None, :, :]
def pred_1m(seq, model):
    pred = model(seq.transpose(1, 2))
    return pred
    
#change h1esc_1m to hff_1m if you want to use HFF model 
pred = pred_1m(torch.FloatTensor(sequence).cuda(), h1esc_1m).squeeze().detach().cpu().numpy()
```


### Train Orca models

If you have set up Orca with its dependencies and has the necessary GPU resources (we have only done training on 4x V100 32Gb servers), you can train new models following the example code under the `train` directory to train new Orca models.  

For training Orca models,  you will need at minimum only a mcool file processed to multiple resolutions including at least 1000, 4000, and, 32000; you can use `cooler zoomify`).  Each model is trained in three stages (1Mb module, 1-32Mb module, 32-256Mb module), corresponding to the three python files. You can refer to the `train/README.md` file for more details of training.  Note that the training process does require significant computing resources and time.

### Q&A
- Error when running the code in CPU mode?
If you have a cuda-capable device but choose to run in the CPU mode,  this can generate a error from torch.DataParallel module. You can fix this by running the code with ```CUDA_VISIBLE_DEVICES= python ...```

- Which datasets are Orca trained on?
You can access these datasets with 4DN accession IDs 4DNFI9GMP2J8 (H1-ESC;  Krietenstein et al. 2020), 4DNFI643OYP9 (HFF; Krietenstein et al. 2020),  4DNFILP99QJS (cohesin-depleted HCT116; Rao et al. 2017). We thank the authors of these high-quality datasets.

- How much resource do I need to run the model locally?
It depends on your application. If you only need predictions for a few variants,  or if you only need the 1Mb model, CPU may be enough. If you need to predict more than a couple dozens of variants, then GPU is highly recommended (>8GB RAM needed).



### Questions and feedbacks
Thank you very much for using Orca. If you have any question or feedback, you can let us know at the [orca-users Google group](https://groups.google.com/g/orca-users). If you found a bug, you can file a Github issue with details for reproducing the bug. Orca is completely free for any non-commercial and academic use, please contact us for other applications.
