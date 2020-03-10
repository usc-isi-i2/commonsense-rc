# TriAN - adaptation for the mowgli framework

Forked from [https://github.com/intfloat/commonsense-rc](https://github.com/intfloat/commonsense-rc).

## Model Overview

We use attention-based LSTM networks.

For more technical details,
please refer to our paper at [https://arxiv.org/abs/1803.00191](https://arxiv.org/abs/1803.00191)

For more details about this task,
please refer to paper [SemEval-2018 Task 11: Machine Comprehension Using Commonsense Knowledge](http://aclweb.org/anthology/S18-1119).

Official leaderboard is available at [https://competitions.codalab.org/competitions/17184#results](https://competitions.codalab.org/competitions/17184#results) (Evaluation Phase)

The overall model architecture is shown below:

![Three-way Attentive Networks](image/TriAN.jpg)

## How to run

Note: Won't work for >= python3.7 due to `async` keyword conflict.

Note: run this in a virtual environment.

Note: GPU machine is preferred, training on CPU will be much slower.

### Setup

* Run `pip install -r requirements.txt`
* `python -m spacy download en`


### Step 1: Prepare data
* Download preprocessed data from [Google Drive](https://drive.google.com/open?id=1M1saVYk-4Xh0Y0Ok6e8liDLnElnGc0P4) or [Baidu Cloud Disk](https://pan.baidu.com/s/1kWHj2z9), unzip and put them under folder data/.
* If you choose to preprocess dataset by yourself,
please run `./download.sh` to download [Glove embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and [ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads), and then run `./run.sh` to preprocess dataset and train the model.

Note: Official dataset can be downloaded on [hidrive](https://my.hidrive.com/lnk/DhAhE8B5).

Note: We transform original XML format data to Json format with [xml2json](https://github.com/hay/xml2json) by running `./xml2json.py --pretty --strip_text -t xml2json -o test-data.json test-data.xml`

### Step 2: Prepare model

* Train model with `python3 src/main.py --gpu 0`,
the accuracy on development set will be approximately 83% after 50 epochs.

## How to reproduce our competition results

Following above instructions you will get a model with ~81.5% accuracy on test set,
we use two additional techniques for our official submission (~83.95% accuracy):

1. Pretrain our model with [RACE dataset](http://www.cs.cmu.edu/~glai1/data/race/) for 10 epochs.

2. Train 9 models with different random seeds and ensemble their outputs.
