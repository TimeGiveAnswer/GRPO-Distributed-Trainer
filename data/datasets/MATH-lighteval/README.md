---
annotations_creators:
- expert-generated
language_creators:
- expert-generated
pretty_name: Mathematics Aptitude Test of Heuristics (MATH)
size_categories:
- 10K<n<100K
source_datasets:
- hendrycks/competition_math
license: mit
dataset_info:
- config_name: algebra
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 955021
    num_examples: 1744
  - name: test
    num_bytes: 648291
    num_examples: 1187
  download_size: 858300
  dataset_size: 1603312
- config_name: counting_and_probability
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 667385
    num_examples: 771
  - name: test
    num_bytes: 353803
    num_examples: 474
  download_size: 504386
  dataset_size: 1021188
- config_name: default
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 5984772
    num_examples: 7500
  - name: test
    num_bytes: 3732833
    num_examples: 5000
  download_size: 4848021
  dataset_size: 9717605
- config_name: geometry
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 1077241
    num_examples: 870
  - name: test
    num_bytes: 523126
    num_examples: 479
  download_size: 813223
  dataset_size: 1600367
- config_name: intermediate_algebra
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 1157476
    num_examples: 1295
  - name: test
    num_bytes: 795070
    num_examples: 903
  download_size: 969951
  dataset_size: 1952546
- config_name: number_theory
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 595793
    num_examples: 869
  - name: test
    num_bytes: 349455
    num_examples: 540
  download_size: 490656
  dataset_size: 945248
- config_name: prealgebra
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 715611
    num_examples: 1205
  - name: test
    num_bytes: 510195
    num_examples: 871
  download_size: 651355
  dataset_size: 1225806
- config_name: precalculus
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: train
    num_bytes: 816245
    num_examples: 746
  - name: test
    num_bytes: 552893
    num_examples: 546
  download_size: 595986
  dataset_size: 1369138
configs:
- config_name: algebra
  data_files:
  - split: train
    path: algebra/train-*
  - split: test
    path: algebra/test-*
- config_name: counting_and_probability
  data_files:
  - split: train
    path: counting_and_probability/train-*
  - split: test
    path: counting_and_probability/test-*
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
- config_name: geometry
  data_files:
  - split: train
    path: geometry/train-*
  - split: test
    path: geometry/test-*
- config_name: intermediate_algebra
  data_files:
  - split: train
    path: intermediate_algebra/train-*
  - split: test
    path: intermediate_algebra/test-*
- config_name: number_theory
  data_files:
  - split: train
    path: number_theory/train-*
  - split: test
    path: number_theory/test-*
- config_name: prealgebra
  data_files:
  - split: train
    path: prealgebra/train-*
  - split: test
    path: prealgebra/test-*
- config_name: precalculus
  data_files:
  - split: train
    path: precalculus/train-*
  - split: test
    path: precalculus/test-*
language:
- en
tags:
- explanation-generation
task_categories:
- text2text-generation
---

# Dataset Card for Mathematics Aptitude Test of Heuristics (MATH) dataset in lighteval format

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
  - [Builder configs](#builder-configs)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://github.com/hendrycks/math
- **Repository:** https://github.com/hendrycks/math
- **Paper:** https://arxiv.org/pdf/2103.03874.pdf
- **Leaderboard:** N/A
- **Point of Contact:** Dan Hendrycks

### Dataset Summary

The Mathematics Aptitude Test of Heuristics (MATH) dataset consists of problems
from mathematics competitions, including the AMC 10, AMC 12, AIME, and more. 
Each problem in MATH has a full step-by-step solution, which can be used to teach
models to generate answer derivations and explanations. This version of the dataset
contains appropriate builder configs s.t. it can be used as a drop-in replacement
for the inexplicably missing `lighteval/MATH` dataset.

## Dataset Structure

### Data Instances

A data instance consists of a competition math problem and its step-by-step solution written in LaTeX and natural language. The step-by-step solution contains the final answer enclosed in LaTeX's `\boxed` tag.

An example from the dataset is:
```
{'problem': 'A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.',
 'level': 'Level 1',
 'type': 'Counting & Probability',
 'solution': 'The spinner is guaranteed to land on exactly one of the three regions, so we know that the sum of the probabilities of it landing in each region will be 1. If we let the probability of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, from which we have $x=\\boxed{\\frac{1}{4}}$.'}
```

### Data Fields

* `problem`: The competition math problem.
* `solution`: The step-by-step solution.
* `level`: The problem's difficulty level from 'Level 1' to 'Level 5', where a subject's easiest problems for humans are assigned to 'Level 1' and a subject's hardest problems are assigned to 'Level 5'.
* `type`: The subject of the problem: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra and Precalculus.

### Data Splits

* train: 7,500 examples
* test: 5,000 examples

### Builder Configs

* default: 7,500 train and 5,000 test examples (full dataset)
* algebra: 1,744 train and 1,187 test examples
* counting_and_probability: 771 train and 474 test examples
* geometry: 870 train 479 test examples
* intermediate_algebra: 1,295 train and 903 test examples
* number_theory: 869 train and 540 test examples
* prealgebra: 1,205 train and 871 test examples
* precalculus: 746 train and 546 test examples

## Additional Information

### Licensing Information

https://github.com/hendrycks/math/blob/main/LICENSE

This repository was created from the [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) dataset. All credit goes to the original authors.

### Citation Information
```bibtex
@article{hendrycksmath2021,
    title={Measuring Mathematical Problem Solving With the MATH Dataset},
    author={Dan Hendrycks
    and Collin Burns
    and Saurav Kadavath
    and Akul Arora
    and Steven Basart
    and Eric Tang
    and Dawn Song
    and Jacob Steinhardt},
    journal={arXiv preprint arXiv:2103.03874},
    year={2021}
}
```

### Contributions

Thanks to [@hacobe](https://github.com/hacobe) for adding this dataset.