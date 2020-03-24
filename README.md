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

### Step 0: Setup

1. Create conda virtual environment and activate it 
`conda create -n cs-rc python=3.6 anaconda` 
`conda actuvate cs-rc`

2. Run `pip install -r requirements.txt`

3. `python -m spacy download en`

4. Run `./download.sh` to download [Glove embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and CSKG.

### Step 1: Prepare data
* Run `./run.sh` to preprocess dataset and train the model.

Note: Official dataset can be downloaded on [hidrive](https://my.hidrive.com/lnk/DhAhE8B5).

Note: We transform original XML format data to Json format with [xml2json](https://github.com/hay/xml2json) by running `./xml2json.py --pretty --strip_text -t xml2json -o test-data.json test-data.xml`

### Step 2: Prepare model

* Train model with `python3 src/main.py --gpu 0`,
the accuracy on development set will be approximately 83% after 50 epochs.

### Notes:

1. Won't work for >= python3.7 due to `async` keyword conflict.
2. Run this in a virtual environment.
3. GPU machine is preferred, training on CPU will be much slower.
