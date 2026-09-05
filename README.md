# Bringing BERT to the field: Transformer models for gene expression prediction in maize

Predicting gene expression levels from upstream promoter regions using deep learning.

---

## Directory Setup

**`scripts/`: directory for production code**

- [`0-data-loading-processing/`](https://github.com/gurveervirk/florabert/tree/master/scripts/0-data-loading-processing):
  - [`01-gene-expression.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/01-gene-expression.py): downloads and processes gene expression data and saves into "B73_genex.txt", may be skipped
  - [`02-download-process-db-data.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/02-download-process-db-data.py): downloads and processes gene sequences from a specified database: 'Ensembl', 'Maize', 'Maize_addition', 'Refseq', consider as the actual first step
  - [`03-combine-databases.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/03-combine-databases.py): combines all the downloaded sequences within all the databases, to be executed after obtaining all processed sequences using [`02-download-process-db-data.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/02-download-process-db-data.py) from the db's
  - [`04-process-genex-nam.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/04-process-genex-nam.py): Pipeline task for processing gene expression data for NAM lines, splitting data for transformer modelling, to be executed after obtaining data from usegalaxy web and [`06-train-test-split.py`](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/06-train-test-split.py)
  - [`05-cluster-maize_seq.sh`](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/05-cluster-maize_seq.sh): clusters the promoter sequences into groups with up to 80% sequence identity, which may be interpreted as paralogs, to be executed after [`03-combine-databases.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/03-combine-databases.py) on Maize NAM data
  - [`06-train-test-split.py`](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/06-train-test-split.py): divides the promoter sequences into train and test sets, avoiding a set of pairs that indicate close relations ("paralogs"), to be executed after [`05-cluster-maize_seq.sh`](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/05-cluster-maize_seq.sh)
  - [`07_train_tokenizer.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/0-data-loading-processing/07_train_tokenizer.py): training byte-level BPE for RoBERTa model, combine all the sequences (excluding Maize), train-test split the combined data (not related to 05-06-04 steps mentioned above, use data from 03) and use train split for training the tokenizer
  - [`08-train-preprocessor.py`](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/08-train-preprocessor.py): Training custom preprocessor for gene expression prediction, may be skipped if no unique preprocessor is reqd
  - [`09-prepare-nam_metadata.py`](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/09-prepare-nam_metadata.py): Preparing cultivar and mazize subpopulation data, can be created by obtaining data from internet directly
- [`1-modeling/`](https://github.com/gurveervirk/florabert/tree/master/scripts/1-modeling)
  - [`pretrain.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/1-modeling/pretrain.py): training the FLORABERT base using a masked language modeling task. Type `python scripts/1-modeling/pretrain.py --help` to see command line options, including choice of dataset and whether to warmstart from a partially trained model. Note: not all options will be used by this script.
  - [`finetune.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/1-modeling/finetune.py): training the FLORABERT regression model (including newly initialized regression head) on multitask regression for gene expression in all 10 tissues. Type `python scripts/1-modeling/finetune.py --help` to see command line options; mainly for specifying data inputs and output directory for saving model weights.
  - [`evaluate.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/1-modeling/evaluate.py): computing metrics for the trained FLORABERT model
- [`2-feature-visualization/`](https://github.com/gurveervirk/florabert/tree/master/scripts/2-feature-visualization)
  - [`embedding_vis.py`](https://github.com/gurveervirk/florabert/blob/master/scripts/2-feature-visualization/embedding_vis.py): computing a sample of BERT embeddings for the testing data and saving to a tensorboard log. Can specify how many embeddings to sample with `--num-embeddings XX` where `XX` is the number of embeddings (must be integer).

**`module/`: directory for our customized modules**

- [`module/`](https://github.com/gurveervirk/florabert/tree/master/module/florabert): our main module named `florabert` that packages customized functions
  - [`config.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/config.py): project-wide configuration settings and absolute paths to important directories/files
  - [`dataio.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/dataio.py): utilities for performing I/O operations (reading and writing to/from files)
  - [`gene_db_io.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/gene_db_io.py): helper functions to download and process gene sequences
  - [`metrics.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/metrics.py): functions for evaluating models
  - [`nlp.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/nlp.py): custom classes and functions for working with text/sequences
  - [`training.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/training.py): helper functions that make it easier to train models in PyTorch and with Huggingface's Trainer API, as well as custom optimizers and schedulers
  - [`transformers.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/transformers.py): implementation of RoBERTa model with mean-pooling of final token embeddings, as well as functions for loading and working with Huggingface's transformers library
  - [`utils.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/utils.py): General-purpose functions and code
  - [`visualization.py`](https://github.com/gurveervirk/florabert/blob/master/module/florabert/visualization.py): helper functions to perform random k-mer flip during data processing and make model prediction

### Pretrained models

If you wish to experiment with our pre-trained FLORABERT models, you can find the saved PyTorch models and the Huggingface tokenizer files [here](https://huggingface.co/Gurveer05/FloraBERT)

**Contents**:

- `byte-level-bpe-tokenizer`: Files expected by a Huggingface `transformers.PretrainedTokenizer`
  - `merges.txt`
  - `vocab.txt`
- transformer: Both language models can instantiate any RoBERTa model from Huggingface's `transformers` library. The prediction model should instantiate our custom `RobertaForSequenceClassificationMeanPool` model class
  1. `language-model`: Trained on all plant promoter sequences
  2. `language-model-finetuned`: Further trained on just maize promoter sequences
  3. `prediction-model`: Fine-tuned on the multitask regression problem

---

### Personal Updates on Forked Repo: 

Kindly refer [`gz_link`](https://github.com/gurveervirk/florabert/tree/main/data/raw/gz_link) for the raw data and links used for data collection and preprocessing, [`research_papers`](https://github.com/gurveervirk/florabert/tree/main/research_papers) for the 26 NAM lines and the references used throughout the project and [`config.py`](https://github.com/gurveervirk/florabert/blob/main/module/florabert/config.py) and the corresponding [`config.yaml`](https://github.com/gurveervirk/florabert/blob/main/config.yaml) for the configurations used (the tissues array in config.py correspond to the values in labels col of data). Majority of the execution has been carried out in [`Colab Notebook`](https://colab.research.google.com/drive/1UsBeiMqeT2ntQbuJhmwceBEswSLOzjr1?usp=sharing) and the pinned [`Kaggle Notebook`](https://www.kaggle.com/code/gurveersinghvirk/florabert-2?scriptVersionId=81).

### ModernBERT support

This fork adds **ModernBERT** as an alternative base model (RoBERTa and DNABERT
paths are kept intact). ModernBERT requires `transformers>=4.48,<5.0` (pin the
4.x line — ModernBERT uses a tokenizer with no dedicated class and is loaded via
`PreTrainedTokenizerFast`).

- **Tokenzier**: train the byte-level BPE tokenizer with ModernBERT-style
  special tokens (`[CLS]/[SEP]/[PAD]/[UNK]/[MASK]`), saved as a fast-tokenizer
  directory (`tokenizer.json`):
  `python scripts/0-data-loading-processing/07_train_tokenizer.py --model modernbert`
- **Pretrain** (MLM): `python scripts/1-modeling/pretrain.py --model-name modernbert-lm`
- **Finetune** (multitask gene-expression regression): `python scripts/1-modeling/finetune.py --model-name modernbert-pred-mean-pool`
- **Evaluate**: `python scripts/1-modeling/evaluate.py --model-name modernbert-pred-mean-pool`

The default small architecture (6 layers, 6 heads, hidden 768) mirrors the
original RoBERTa config; `modernbert-base.intermediate_size: 2048` is set so the
GeGLU MLP has roughly the same number of parameters as RoBERTa's 3072-wide MLP,
keeping the head-to-head comparison param-matched. ModernBERT checkpoints are
saved separately under `models/transformer/language-model-modernbert` and
`models/transformer/prediction-model-modernbert`, and use
`models/modernbert-byte-level-bpe-tokenizer`.

A ready-to-run GPU **Kaggle notebook** (tokenizer → pretrain → finetune →
evaluate) is provided at
[`notebooks/modernflorabert_modernbert.ipynb`](notebooks/modernflorabert_modernbert.ipynb).
It **clones this repository for code** and **copies the data from the
`florabert-base` Kaggle dataset** (the original data is kept there; this repo
is intentionally lean and does not ship the raw data or intermediate outputs).

**First module has been completed. All data / outputs are under [`data`](https://github.com/gurveervirk/florabert/tree/main/data) or [`models`](https://github.com/gurveervirk/florabert/tree/main/models). Moving to Second Module. The following steps were essential for this [script](https://github.com/gurveervirk/florabert/blob/main/scripts/0-data-loading-processing/04-process-genex-nam.py).**

The following updates have been done using python scripts under [`3-RNAseq-quantification/`](https://github.com/gurveervirk/florabert/tree/master/scripts/3-RNAseq-quantification):

- The scripts under this module requires a lot of resources and time (patience). We opted to use the Bioinformatics website [Galaxy](https://usegalaxy.org/). This provides every user 250GB storage and allows the ability to use a number of very useful and important bioinformatics tools. 

- The scripts under the module dealt with 26 NAM lines / cultivars of Maize. We replicated the entire process under this module in this website, with some minor changes (not in output). The first step was to get all the runs corresponding to each cultivar and unique organsim part for each, to avoid repitition. 

- This was achieved by getting the base data from [EBI](https://www.ebi.ac.uk/) and searching for the 2 Bioprojects mentioned in the supplementary material (under [`research_papers`](https://github.com/gurveervirk/florabert/tree/main/research_papers)). This data was then used alongside the [`helper_codes`](https://github.com/gurveervirk/florabert/tree/main/helper_codes) scripts to get the file [`unique_orgs_runs.tsv`](https://github.com/gurveervirk/florabert/blob/main/helper_files/unique_orgs_runs.tsv). This file contains the runs corresponding to unique organism parts of each cultivar.

- A workflow was then created / implemented / configured (the base workflow was created by user vasquex11 on the mentioned website) to align with the scripts. The runs were first uploaded per cultivar to the website (after logging in) in txt format, one per line. Next, fasterq-dump tool was used with --split-files option selected to get the fastq files corresponding to the runs.

- The created workflow [`FloraBERT Test (Trimmomatic + HISAT2 + featureCounts)`](https://usegalaxy.org/u/gurveer05/w/copy-of-module-72-part-1-trimmomatic--hisat2--featurecounts-shared-by-user-vasquex11) was used to perform all the actions mentioned in the module. The final output are the featureCounts files corresponding to each run ( extending to unique organsim part of cultivars ). The steps are self-explanatory (using the research papers).

**The full train-test data for pretraining is available at [`Hugging Face Dataset For Pretraining`](https://huggingface.co/datasets/Gurveer05/plant-promoter-sequences), for finetuning on Maize NAM data at [`Hugging Face Dataset For Finetuning`](https://huggingface.co/datasets/Gurveer05/maize-promoter-sequences) and for finetuning for the downstream task at [`Hugging Face Dataset for Downstream task`](https://huggingface.co/datasets/Gurveer05/maize-nam-gene-expression-data)**

**[`Evaluation results`](https://www.kaggle.com/datasets/gurveersinghvirk/florabert-base/versions/83)** are also under [`outputs/`](https://github.com/gurveervirk/florabert/tree/main/output).

**Some observations**:

- using different transformations to handle the **highly right skewed TPM values** (during finetuning stage):
  - natural log transformation gave an mse of 2.4
  - boxcox transformation gave an mse of 12.9
  - **log10 transformation** gave an mse of **0.4441**
  - log10 also improved r2 by 0.01 (from 0.077 to 0.087) which is considerable here (no other changes)
  - this means that handling the highly right skewed is pivotal here for model performance
- TPU usage on pytorch is a herculean task for our use-case
- custom finetuning script might need some tweaking
- python dependencies are problematic at times
- DNABERT-1 makes use of k-mers meaning the tokenizer files that are made are basically 4^k permutations ('A', 'T', 'G', 'C'), not great (now) for tasks dealing with DNA
- DNABERT-2 is an obvious improvement as it makes use of actual tokenization, preventing overlapping and better identification of important parts of DNA seqeunces
- Byte-level-BPE-tokenizer makes use of 256 basic (byte-level) tokens; taking this into consideration, new vocab_size is set to 5256
- **padding HAS TO true OR "max_length" WHEN TRAINING ON TPU** (as a parameter for tokenizer; used in preprocess_fn in dataio.py in this proj)
- torch, torch_xla, torch_xla.core.xla_model (as xm) have to be imported to make sure Training (Trainer) works properly on TPU (tested on kaggle)

**Citations**:
- *Base Papers for [**FloraBERT**](https://github.com/benlevyx/florabert/tree/main) are available under [`research_papers`](https://github.com/gurveervirk/florabert/tree/main/research_papers)*
- The [Galaxy](https://usegalaxy.org/) Community. [The Galaxy platform for accessible, reproducible and collaborative biomedical analyses: 2022 update](https://academic.oup.com/nar/article/50/W1/W345/6572001), *Nucleic Acids Research*, Volume 50, Issue W1, 5 July 2022, Pages W345–W351, doi:10.1093/nar/gkac247
- National Center for Biotechnology Information (NCBI)[Internet]. Bethesda (MD): National Library of Medicine (US), National Center for Biotechnology Information; [1988] – [cited 2024 Jan 26]. Available from: https://www.ncbi.nlm.nih.gov/
- [Project: PRJEB35943 at EBI](https://www.ebi.ac.uk/ena/browser/view/PRJEB35943)
- [Project: PRJEB36014 at EBI](https://www.ebi.ac.uk/ena/browser/view/PRJEB36014)
- Andrew D. Yates, James Allen, Ridwan M. Amode, Andrey G. Azov, Matthieu Barba, Andrés Becerra, Jyothish Bhai, Lahcen I. Campbell, Manuel Carbajo Martinez, Marc Chakiachvili, Kapeel Chougule, Mikkel Christensen, Bruno Contreras-Moreira, Alayne Cuzick, Luca Da Rin Fioretto, Paul Davis, Nishadi H. De Silva, Stavros Diamantakis, Sarah Dyer, Justin Elser, Carla V. Filippi, Astrid Gall, Dionysios Grigoriadis, Cristina Guijarro-Clarke, Parul Gupta, Kim E. Hammond-Kosack, Kevin L. Howe, Pankaj Jaiswal, Vinay Kaikala, Vivek Kumar, Sunita Kumari, Nick Langridge, Tuan Le, Manuel Luypaert, Gareth L. Maslen, Thomas Maurel, Benjamin Moore, Matthieu Muffato, Aleena Mushtaq, Guy Naamati, Sushma Naithani, Andrew Olson, Anne Parker, Michael Paulini, Helder Pedro, Emily Perry, Justin Preece, Mark Quinton-Tulloch, Faye Rodgers, Marc Rosello, Magali Ruffier, James Seager, Vasily Sitnik, Michal Szpak, John Tate, Marcela K. Tello-Ruiz, Stephen J. Trevanion, Martin Urban, Doreen Ware, Sharon Wei, Gary Williams, Andrea Winterbottom, Magdalena Zarowiecki, Robert D. Finn and Paul Flicek. **[Ensembl Genomes](https://plants.ensembl.org/index.html) 2022: an expanding genome resource for non-vertebrates.**, *Nucleic Acids Research* 2022, https://doi.org/10.1093/nar/gkab1007
- [MaizeGDB](https://www.maizegdb.org):  Woodhouse MR, Cannon EK, Portwood JL, Harper LC, Gardiner JM, Schaeffer ML, Andorf CM. (2021) A pan-genomic approach to genome databases using maize as a model system. BMC Plant Biol 21, 385. doi: https://doi.org/10.1186/s12870-021-03173-5.
- O'Leary NA, Wright MW, Brister JR, Ciufo S, Haddad D, McVeigh R, Rajput B, Robbertse B, Smith-White B, Ako-Adjei D, Astashyn A, Badretdin A, Bao Y, Blinkova O, Brover V, Chetvernin V, Choi J, Cox E, Ermolaeva O, Farrell CM, Goldfarb T, Gupta T, Haft D, Hatcher E, Hlavina W, Joardar VS, Kodali VK, Li W, Maglott D, Masterson P, McGarvey KM, Murphy MR, O'Neill K, Pujar S, Rangwala SH, Rausch D, Riddick LD, Schoch C, Shkeda A, Storz SS, Sun H, Thibaud-Nissen F, Tolstoy I, Tully RE, Vatsan AR, Wallin C, Webb D, Wu W, Landrum MJ, Kimchi A, Tatusova T, DiCuccio M, Kitts P, Murphy TD, Pruitt KD. **Reference sequence [RefSeq](https://www.ncbi.nlm.nih.gov/refseq/) database at NCBI: current status, taxonomic expansion, and functional annotation.** *Nucleic Acids Res*. 2016 Jan 4;44(D1):D733-45 PubMed PubMedCentral
