## Download dataset

- 2012-2016 from https://github.com/brmson/dataset-sts/tree/master/data/sts/semeval-sts
- 2017 input from http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.eval.v1.1.zip, ground truth from http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.gs.zip

## Arrange files

```
│  create_dataset.ipynb
├─2012
│      MSRpar.test.tsv
│      MSRpar.train.tsv
│      OnWN.test.tsv
│      SMTeuroparl.test.tsv
│      SMTeuroparl.train.tsv
│      SMTnews.test.tsv
│
├─2013
│      correct-output.pl
│      FNWN.test.tsv
│      headlines.test.tsv
│      OnWN.test.tsv
│
├─2014
│      deft-forum.test.tsv
│      deft-news.test.tsv
│      headlines.test.tsv
│      images.test.tsv
│      OnWN.test.tsv
│      tweet-news.test.tsv
│
├─2015
│      answers-forums.test.tsv
│      answers-students.test.tsv
│      belief.test.tsv
│      headlines.test.tsv
│      images.test.tsv
│
├─2016
│      answer-answer.test.tsv
│      headlines.test.tsv
│      plagiarism.test.tsv
│      postediting.test.tsv
│      question-question.test.tsv
│
└─2017
        STS.gs.track5.en-en.txt
        STS.input.track5.en-en.txt
```

## Create dataset

Run all commands in `create_dataset.ipynb`, you will get two files: `train.json` and `valid.json`.
