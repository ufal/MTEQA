# MTEQA

This repository is the official implementation
of [Just Ask! Evaluating Machine Translation by Asking and Answering Questions](http://www.statmt.org/wmt21/pdf/2021.wmt-1.58.pdf).

> ![MTEQA-basic](./resources/MTEQA-basic.png?raw=true "Title")

## Requirements

Our implementation is based on the code and models provided
in: [question_generation](https://github.com/patil-suraj/question_generation).

To clone this repo, run:

```setup
git clone https://github.com/ufal/MTEQA
cd MTEQA
git submodule init
git submodule update
```

or just:

```setup
git clone --recurse-submodules https://github.com/ufal/MTEQA
cd MTEQA
```

and then install required dependencies and download required models with:

```setup
pip install -r requirements.txt
./mteqa/download_models.py
```

**Code was tested with python 3.8.**

## Scoring

To score the MT output, run

```eval
python mteqa_score.py --reference ref --hypothesis out --lang en > mteqa_score.tsv
```

Essential arguments are:

* `reference`: Path to the file with reference translations, one segment per line.
* `hypothesis`: Path to the file with MT system output, one segment per line.
* `lang`: ISO code of the target language **(23.11.2021 - only English [en] is supported)**.

Additional flags are:

* `--cpu`: Force interference on CPU, by default GPU is used if detected.
* `--baseline_qe`: Use the baseline system for Answer Extraction,
  see [Section 3.3 - 2)](http://www.statmt.org/wmt21/pdf/2021.wmt-1.58.pdf). By default keyword extraction based on POS
  pattern matching/NER is used.
* `--gen_from_out`: Extract Question/Answer pairs from both the reference and the MT output,
  see [Section 3.3 - 1)](http://www.statmt.org/wmt21/pdf/2021.wmt-1.58.pdf).
* `--verbose`: Outputs per-segment score. By default, only the single system-level score is reported.

Output is tab delimited, with a single column for each string-comparison metric that we used.

To reproduce [our results](http://www.statmt.org/wmt21/pdf/2021.wmt-1.110.pdf) from the WMT 2021 Metrics Shared Task you
should use the default parameters (i.e. POS/NER based keyword extraction, questions generated only from reference) and
consider the **chrf** metric for answer comparison.
