# Applied AI Miniproject -- Group 16
*Author: Florian LACHAUX (flolac-3@student.ltu.se)*


## ----------------------------

## Program dependencies

Made on Python 3.9 with the following libraries:
- torch (v2.1.2)
- torchvision (v0.16.2)
- tqdm (v4.66.1)
- matplotlib (v3.8.2)

## Problem & data description

**Category:** Supervised Learning -- Vectorized data -- Classification

I wanted to perform model selection for the best classifier based on validation data for the dataset MNIST and the `digits` split of dataset EMNIST.

Both datasets are not included in the repository for space reasons, but they are **downloaded automatically** by the program. They can also be found here:

MNIST: http://yann.lecun.com/exdb/mnist/
EMNIST: http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip

Both datasets are pre-processed to be tensorized and normalized between 0 and 1.

## Models & Program behavior

I initially wanted to have both Multi-Layer Perceptrons and CNNs available to train and test, but I only had the time to do MLP.
The MLP is very modular, allowing for any number and sizes of hidden layers, and three different activation functions (ReLU, Sigmoid, TanH).
Its output layer is a LogSoftmax probability function, and it's used in conjuction with a negative-log-likelihood loss in training.

The program allows the user to define multiple MLP models, and then it trains and validates them all on the selected dataset. The best model is selected as the one with the smallest validation loss on the last training epoch.
It then tests the best classifier model on the testing data, and obtains the final testing loss and estimated classifier risk (unbiased likelihood of classification error).

The program has been proven to work with GPU parallelization using CUDA 11.7, the batch size needs to be increased to see the significant performance improvements.

**How to run:** Run script main.py with Python with all required modules installed.

## Video demonstration

Watch Youtube Video:
https://youtu.be/2OTh_bMTNs4

## -------------------------------

## Test Results

On both datasets MNIST and EMNIST (digits), I recorded the performance of 15 models on 20 epochs and selected the best :

| Model  | Layers | Activation function |
| ------------- | ------------- | ------------- |
| 1 | [784, 32, 10] | ReLU |
| 2 | [784, 32, 10] | Sigmoid |
| 3 | [784, 64, 10] | ReLU |
| 4 | [784, 64, 10] | Sigmoid |
| 5 | [784, 128, 10] | ReLU |
| 6 | [784, 32, 10] | Sigmoid |
| 7 | [784, 64, 32, 10] | ReLU |
| 8 | [784, 128, 64, 10] | ReLU |
| 9 | [784, 128, 64, 32, 10] | ReLU |
| 10 | [784, 64, 32, 16, 10] | ReLU |
| 11 | [784, 128, 64, 32, 16, 10] | ReLU |
| 12 | [784, 64, 64, 64, 16, 10] | ReLU |
| 13 | [784, 64, 32, 64, 32, 10] | ReLU |
| 14 | [784, 80, 60, 40, 20, 10] | ReLU |
| 15 | [784, 200, 200, 200, 200, 10] | ReLU |

In every test, I have taken apart 20% validation data in the training set.

#### Performance on MNIST

**Training loss:**

| Epoch ------ Model | 1<br><br><br> | 2<br><br><br> | 3<br><br><br> | 5<br><br><br> | 10<br><br><br> | 15<br><br><br> | 20<br><br><br> |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| 1 | 0.49338 | 0.33702 | 0.30862 | 0.29216 | 0.26115 | 0.24565 | 0.24325 |
| 2 | 0.51924 | 0.27491 | 0.23393 | 0.19566 | 0.15679 | 0.14016 | 0.12422 |
| 3 | 0.38308 | 0.22071 | 0.18501 | 0.14895 | 0.11566 | 0.10061 | 0.08669 |
| 4 | 0.43466 | 0.23264 | 0.19489 | 0.19613 | 0.13159 | 0.12202 | 0.11184 |
| 5 | 0.35523 | 0.18926 | 0.15535 | 0.13026 | 0.09646 | 0.07862 | 0.07670 |
| 6 | 0.37596 | 0.20618 | 0.17287 | 0.14389 | 0.11725 | 0.10304 | 0.09517 |
| 7 | 0.40281 | 0.23063 | 0.18687 | 0.15355 | 0.11810 | 0.09899 | 0.08670 |
| 8 | 0.34532 | 0.18351 | 0.15035 | 0.12166 | 0.09321 | 0.07601 | 0.06609 |
| 9 | 0.41151 | 0.19746 | 0.16175 | 0.12829 | 0.08939 | 0.07422 | 0.06566 |
| 10 | 0.46355 | 0.22827 | 0.18840 | 0.16164 | 0.11668 | 0.10146 | 0.09396 |
| 11 | 0.49400 | 0.22032 | 0.18063 | 0.13693 | 0.10155 | 0.08279 | 0.07210 |
| 12 | 0.49461 | 0.23698 | 0.19224 | 0.15747 | 0.11690 | 0.09422 | 0.08779 |
| 13 | 0.52182 | 0.26569 | 0.21534 | 0.17018 | 0.13004 | 0.11558 | 0.10534 |
| 14 | 0.50870 | 0.23437 | 0.19944 | 0.15700 | 0.11113 | 0.10342 | 0.11621 |
| 15 | 0.40471 | 0.21513 | 0.17110 | 0.13756 | 0.10251 | 0.08109 | 0.07103 |

**Validation loss:**

| Epoch ------ Model | 1<br><br><br> | 2<br><br><br> | 3<br><br><br> | 5<br><br><br> | 10<br><br><br> | 15<br><br><br> | 20<br><br><br> |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| 1 | 0.35671 | 0.30273 | 0.41227 | 0.29106 | 0.27689 | 0.26883 | 0.27631 |
| 2 | 0.34065 | 0.26130 | 0.24846 | 0.22455 | 0.18810 | 0.19144 | 0.19932 |
| 3 | 0.29609 | 0.22004 | 0.18376 | 0.16868 | 0.13884 | 0.15902 | 0.17860 |
| 4 | 0.26718 | 0.20724 | 0.19308 | 0.20984 | 0.16726 | 0.17606 | 0.19122 |
| 5 | 0.21078 | 0.18680 | **0.14434** | **0.14430** | 0.16694 | 0.16594 | 0.19851 |
| 6 | 0.22547 | 0.19257 | 0.17097 | 0.17779 | 0.13842 | 0.15065 | 0.13693 |
| 7 | 0.31178 | 0.20530 | 0.20948 | 0.15311 | 0.14975 | 0.16493 | 0.17294 |
| 8 | **0.18095** | **0.16940** | 0.17750 | 0.16231 | 0.13476 | **0.14009** | 0.19701 |
| 9 | 0.31514 | 0.17445 | 0.17525 | 0.15021 | 0.14259 | 0.14293 | 0.17203 |
| 10 | 0.27080 | 0.21802 | 0.38321 | 0.17701 | 0.14253 | 0.17409 | 0.15199 |
| 11 | 0.27508 | 0.19529 | 0.19654 | 0.14599 | **0.13167** | 0.14055 | **0.12077** |
| 12 | 0.28660 | 0.21113 | 0.20167 | 0.15645 | 0.16556 | 0.14537 | 0.18404 |
| 13 | 0.36138 | 0.27484 | 0.22530 | 0.19274 | 0.16678 | 0.17931 | 0.21851 |
| 14 | 0.26244 | 0.35728 | 0.16352 | 0.15162 | 0.14062 | 0.16579 | 0.13098 |
| 15 | 0.24371 | 0.20726 | 0.24082 | 0.17485 | 0.18285 | 0.15221 | 0.17805 |

The best model selected was **Model 11** because it had the lowest validation loss at the last epoch.

**Best Model Performance:**

Evolution of training and validation losses:
![](https://www.pixenli.com/image/NbL46I5t)
![](https://www.pixenli.com/image/F1orrXLM)

Testing loss on full test set: 0.11582
Estimated classifier risk: **0.0286**


#### Performance on EMNIST (digits)

**Training loss:**

| Epoch ------ Model | 1<br><br><br> | 2<br><br><br> | 3<br><br><br> | 5<br><br><br> | 10<br><br><br> | 15<br><br><br> | 20<br><br><br> |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| 1 | 0.23225 | 0.16084 | 0.14470 | 0.13095 | 0.11926 | 0.11432 | 0.11203 |
| 2 | 0.25032 | 0.15617 | 0.14298 | 0.13011 | 0.11499 | 0.11070 | 0.10662 |
| 3 | 0.19196 | 0.12182 | 0.10905 | 0.09773 | 0.09015 | 0.08414 | 0.08113 |
| 4 | 0.20687 | 0.12951 | 0.11488 | 0.10624 | 0.09085 | 0.08383 | 0.07939 |
| 5 | 0.17031 | 0.10956 | 0.09708 | 0.08783 | 0.07458 | 0.07137 | 0.06638 |
| 6 | 0.18013 | 0.11516 | 0.10165 | 0.09620 | 0.08935 | 0.08089 | 0.07714 |
| 7 | 0.18353 | 0.10921 | 0.09582 | 0.08180 | 0.06857 | 0.06401 | 0.06102 |
| 8 | 0.17031 | 0.10225 | 0.09141 | 0.07946 | 0.06368 | 0.06001 | 0.05762 |
| 9 | 0.17870 | 0.10158 | 0.08786 | 0.07405 | 0.05830 | 0.05274 | 0.04919 |
| 10 | 0.21166 | 0.12709 | 0.10969 | 0.09338 | 0.07937 | 0.07294 | 0.06839 |
| 11 | 0.21209 | 0.10568 | 0.08991 | 0.07560 | 0.06007 | 0.05271 | 0.04621 |
| 12 | 0.22197 | 0.12201 | 0.10612 | 0.09181 | 0.07656 | 0.07035 | 0.06689 |
| 13 | 0.22251 | 0.13363 | 0.11429 | 0.09731 | 0.08395 | 0.08157 | 0.07502 |
| 14 | 0.21970 | 0.11698 | 0.10077 | 0.08201 | 0.06445 | 0.05941 | 0.05581 |
| 15 | 0.19446 | 0.11215 | 0.09893 | 0.08141 | 0.06869 | 0.06688 | 0.05811 |

**Validation loss:**

| Epoch ------ Model | 1<br><br><br> | 2<br><br><br> | 3<br><br><br> | 5<br><br><br> | 10<br><br><br> | 15<br><br><br> | 20<br><br><br> |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| 1 | 0.16698 | 0.15356 | 0.14519 | 0.12879 | 0.13030 | 0.12806 | 0.13487 |
| 2 | 0.17376 | 0.15195 | 0.15211 | 0.15272 | 0.12323 | 0.12539 | 0.11711 |
| 3 | 0.13847 | 0.12475 | 0.13147 | 0.11423 | 0.10350 | 0.12176 | 0.11516 |
| 4 | 0.14430 | 0.14916 | 0.12821 | 0.11515 | 0.10690 | 0.09822 | 0.09998 |
| 5 | 0.13783 | 0.10808 | 0.12706 | 0.09356 | 0.10152 | 0.10130 | 0.12530 |
| 6 | 0.13403 | 0.10832 | 0.10292 | 0.09911 | 0.10010 | 0.09174 | 0.08553 |
| 7 | 0.11786 | 0.09863 | 0.10111 | 0.08824 | 0.09125 | 0.09269 | 0.08642 |
| 8 | **0.10202** | 0.12578 | 0.09991 | 0.08104 | 0.09333 | 0.08879 | 0.09245 |
| 9 | 0.11199 | **0.09552** | **0.08724** | 0.09690 | **0.07462** | **0.07227** | **0.07814** |
| 10 | 0.20927 | 0.14090 | 0.10029 | 0.09911 | 0.10236 | 0.08471 | 0.08782 |
| 11 | 0.12130 | 0.11057 | 0.10221 | **0.07664** | 0.08839 | 0.08705 | 0.08548 |
| 12 | 0.12869 | 0.11576 | 0.10888 | 0.11777 | 0.10054 | 0.08850 | 0.09281 |
| 13 | 0.15221 | 0.12728 | 0.12805 | 0.11039 | 0.09848 | 0.09896 | 0.10005 |
| 14 | 0.11614 | 0.14648 | 0.09053 | 0.09058 | 0.07769 | 0.07919 | 0.08299 |
| 15 | 0.12269 | 0.11956 | 0.11652 | 0.08710 | 0.08792 | 0.09187 | 0.15090 |

The best model selected was **Model 9** because it had the lowest validation loss at the last epoch.

**Best Model Performance:**

Evolution of training and validation losses:
![](https://www.pixenli.com/image/AC6nsI3z)
![](https://www.pixenli.com/image/rY6MOJC3)

Testing loss on full test set: 0.07041
Estimated classifier risk: **0.0157**