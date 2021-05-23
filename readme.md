# Code for SupMMD

This document provide a brief overview of code for SupMMD paper.

The steps are roughly:

## 1. Preprocessing
DUC (03/04) and TAC(08/09) datasets are available from [their website](https://duc.nist.gov/data.html).
We provide code for preprocessing the datasets.
We use the proprocessing on DUC, for TAC, we use the [preprocessing code from ICSI](https://github.com/benob/icsisumm). But our preprocessing code works for TAC as well.

1. `scripts/parse_data.py duc|tac` to preprocess the XML files into a cleaned json file. We provide the cleaned json file for DUC and TAC(cleaned by ICSI) inside `data_preprocessed.tar.gz` for reviewers to not going throuogh the process of requesting data from DUC/TAC.
2. `scripts/punkt_sent.py` to learn PunktSentenceTokenizer from NLTK. It requires the cleaned json files. The tokenizer is saved inside `commons/` directory.
3. To extract DBPedia spotlight concepts: `data/entities.py`. It requires the cleaned json files from step 1. It requires locally installing DBPedia spotlight. We provided the extracted entities files inside `data_preprocessed.tar.gz` to avoid installion process.

## 2. Oracle Extraction
We extract the oracles using `oracle/oracle.py` for DUC and `oracle/oracle4icsi.py` for TAC datasets.
This will create the csv files with each sentence in a row, with few surface features such as #words, #nouns and oracle labels (y_i = 1 if i in oracle summary, 0 else). We provide the csv files inside [`data_preprocessed.tar.gz`](https://www.dropbox.com/s/uxxgx684fojinrs/data_preprocessed.tgz?dl=0).

After oracle extraction, we extract keywords (although we don't use it in the paper) using `data/keywords.py`. The extracted keywords json files is inside [`data_preprocessed.tar.gz`](https://www.dropbox.com/s/uxxgx684fojinrs/data_preprocessed.tgz?dl=0).

## 4. MKL weights

First, we cache the datasets for each dataset in `sup_mmd/data.py` class object.
```
	data.py duc03 0 y_hm_0.4 ## to parse duc03 with our oracle method oracles
	data.py tac09 0 y_R2_0.0 ## to parse tac09 with oracle method from Liu & Lapata 2019
```

We suggest making a `project_dir` where there is some space  (~ 5GB, as there will be lots of summary files created). And keep all data jsons, csv files as well as `.pik` cache files from this step inside `project_dir/data`.

Then, we use `sup_mmd/kernel_align.py` as
```
	kernel_align.py duc03 A y ## for DUC03/4 generic summ weights
	kernel_align.py tac08 A y ## for TAC08/09 generic summ weights
	kernel_align.py tac08 B y ## for TAC08/09 comparative summ weights
```

## 5. Training all models according to hyperparameter grid.
We train several models with combination of hyperparams provided by a conf file in [HOCON format](https://github.com/lightbend/config/blob/master/HOCON.md).
We provide the conf file for grids we use for each dataset inside `sup_mmd/confs/` directory. Please specify the `project_dir` that we created in step 4 in `ROOT` variable. There should be `project_dir/data/` directory with all necessary data files we have created till now. Please change other variables such as `N_PROC` as convinient. We provide following conf files.
1. `confs/duc03.conf`: Train all models using our oracles on DUC03 dataset.
2. `confs/duc03_R2r0.conf`: Train all models using our oracles as Liu & Lapata 2019.
3. And similarly for TAC08A and TAC08B datasets.

We have following two scripts to train all models as specified by grid. This is required because ROUGE package can't be incorporated inside gridsearch frameworks like sklearn's ones. So we train all models, infer summaries for each model, compute ROUGE and keep the best model. The scripts are implemented using python multiprocessing to train several models in parallel.

- `sup_mmd/run_generic.py confs/DUC03.conf` to train all models (generic task) according to provided grids.
- `MODEL=lin1 sup_mmd/run_update.py confs/TAC08.conf` to train all models (comparaive task, single linear model) according to provided grids.
- `MODEL=lin2 sup_mmd/run_update.py confs/TAC08.conf` to train all models (comparaive task, 2 linear models) according to provided grids.

Once the script finishes, it will create a directory for each run like DUC03 or TAC08B_lin1_R2r0. Each of this directory contains 3 directories -- 
- `logs/`: will contain train logs for each model
- `runs/`: will contains tensorboard summary writes
- `states/`: will contains the saved models

## 6. Inference
`python infer.py DUC03`, the argument is the directory created by training over grids of hyperparameters.
This will create a huge number of small text file, each one containing a summary due to one model for one topic.

## 7. ROUGE eval
`python ../rouge/create_rouge.py duc03 summaries_dir` create a XML file to be provided to ROUGE package. We create several of these directory in previous step, so we could use following example bash script to create XML files for DUC03 trained models. Similarly for tac08-A and tac08-B datasets and alt oracles. Please note the first argument which should match the train and test dataset.
```
#for d in project_dir/DUC03/rouge_train/*; do python ../rouge/create_template.py duc03 $d; done
#for d in project_dir/DUC03/rouge_test/*; do python ../rouge/create_template.py duc04 $d; done

```
Once the xml files are created for ROUGE package, to evaluate DUC 
```
	for f in DUC03/rouge_*/*.xml; do ROUGE-1.5.5/ROUGE-1.5.5.pl -n 4 -m -a -l 100 -x -c 95 -r 1000 -f A -p 0.5 -t 0 $f > $f.txt & done
```
to evaluate TAC-A/B,
```
	for f in TAC08A/rouge_*/*.xml; do ROUGE-1.5.5/ROUGE-1.5.5.pl -n 4 -w 1.2 -m  -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -l 100 $f > $f.txt & done
```
Once finished, ROUGE text results are created. Then combine the small files as:
```
cat rouge_test/*.txt > gs_test.txt && cat rouge_train/*.txt > gs_train.txt
```
Then use `./convert2csv.sh DUC03` to convert combined ROUGE text file to nice csv.

## 8.Results
Once we have csv of ROUGE evaluations from previous step. We can use R scripts `eval_generic.R` or `eval_update.R` to analyse the result. It will create `res.R.csv` in each folder specified in the script. The resulting csv have score column, for each alpha (MKL or bigrams kernel), pick the one with highest score. We report these numbers in the paper.

We also generate and eval the summaries due to sentence compression as discussed on paper using `compressed_summary.py model_file 0.001 1`
where the first arg is the path to best model selected by validation as in previous step, and second arg r (parameter of modified greedy algorithm) selected by validation as in previous step. We provide the best model, summaries of best model, along with sentence compression inside the directories `duc04_A`, `tac09_A` and `tac09_B` .

## Misc
- `unsup_generic.py` and `unsup_update.py` for unsupervised MMD summaries generation. We do validation exactly to supervised method.
- `model.py` contains the models, implemented in pytorch
- `run_generic.py`, `run_update.py` trains a number of models provided from hyperparams grid
- `functions.py` contains helper functions

## Dependencies
- python version 3.7. An export of anaconda environment is provided as `environment.yml` file.
