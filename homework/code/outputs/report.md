# Binary Tongue Coating Classification Report

## 1. Objective

This study classifies tongue images into two classes, `coated` and `non_coated`, and follows a progressive experimentation pipeline. The goal is not only to report the final model performance, but also to show how the performance changes when preprocessing, training-only augmentation, feature/model enhancement, and validation-based post-processing are added step by step.

All experiments were implemented in Python and were made reproducible by fixing the random seed. The same train, validation, and test samples were used in every experiment so that the reported before/after comparison is fair.

## 2. Dataset and Reproducibility

The dataset contains 2007 JPG images. There are 1001 coated and 1006 non-coated samples. The random seed was fixed as `561`.

The dataset was split with stratified sampling into 80% training, 10% validation, and 10% testing. The test set was kept unseen during model selection and threshold selection.

| Split | coated | non_coated | Total |
|---|---:|---:|---:|
| train | 801 | 804 | 1605 |
| validation | 100 | 101 | 201 |
| test | 100 | 101 | 201 |

## 3. Neural Network and Experimental Pipeline

The classifier in all experiments was a Multi-Layer Perceptron (MLP) neural network. A `StandardScaler` was fitted only on the training features and then applied to validation and test features. The MLP used ReLU activation, Adam optimizer, batch size 32, early stopping, and an internal validation fraction of 0.15 inside the training split. L2 regularization was controlled by the `alpha` parameter.

The progressive experiments were:

- **E1 baseline:** resized RGB pixels only. This provides a simple reference model.
- **E2 preprocessing:** gray-world color normalization and CLAHE contrast enhancement before pixel features.
- **E3 augmentation:** horizontal flip, small rotation, brightness jitter, and contrast jitter applied only to training images.
- **E4 model enhancement and post-processing:** HOG texture descriptors, HSV/color statistics, training-only augmentation, and a decision threshold selected on the validation set.

The table below gives the main model hyperparameters and feature dimensions.

| ID | Approach | Hidden layers | Alpha | LR | Max iter | Features | Train samples | Threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E1_baseline_raw_pixels | Basic neural network on resized RGB pixels; no enhancement. | (64,) | 0.0001 | 0.001 | 120 | 12288 | 1605 | 0.500 |
| E2_preprocessed_pixels | Same NN input after gray-world color normalization and CLAHE. | (96,) | 0.0001 | 0.0008 | 140 | 12288 | 1605 | 0.500 |
| E3_augmented_preprocessed_pixels | Training-only flips, small rotations, and color jitter on preprocessed pixels. | (128, 64) | 0.0002 | 0.0008 | 160 | 12288 | 6420 | 0.500 |
| E4_hog_color_postprocessed | HOG/color model enhancement with validation-selected decision threshold. | (256, 64) | 0.0005 | 0.0005 | 220 | 1812 | 6420 | 0.200 |

## 4. Results

| Experiment | Split | Accuracy | Precision | F-score |
|---|---|---:|---:|---:|
| E1_baseline_raw_pixels | validation | 0.9204 | 0.9381 | 0.9192 |
| E1_baseline_raw_pixels | test | 0.8806 | 0.9053 | 0.8776 |
| E2_preprocessed_pixels | validation | 0.8955 | 0.9000 | 0.8955 |
| E2_preprocessed_pixels | test | 0.9204 | 0.9474 | 0.9184 |
| E3_augmented_preprocessed_pixels | validation | 0.8955 | 0.9000 | 0.8955 |
| E3_augmented_preprocessed_pixels | test | 0.8955 | 0.9444 | 0.8901 |
| E4_hog_color_postprocessed | validation | 0.9453 | 0.9327 | 0.9463 |
| E4_hog_color_postprocessed | test | 0.9204 | 0.9048 | 0.9223 |

The best test F-score was obtained by `E4_hog_color_postprocessed` with accuracy=0.9204, precision=0.9048, and F-score=0.9223. Compared with the baseline model, this corresponds to changes of +0.0398 accuracy, -0.0005 precision, and +0.0448 F-score.

Preprocessing gave the clearest test-set improvement over the baseline. Augmentation did not improve the test result in this particular split, which suggests that the chosen transformations may have introduced variations that were not fully representative of the unseen samples. The final HOG/color experiment improved the F-score by reducing the number of missed positive samples, although its precision was lower than the preprocessed pixel model.

## 5. Conclusion and Future Work

The experiments show that a systematic pipeline is useful for this binary tongue coating classification problem. Starting from a simple neural network on raw resized pixels, color and contrast preprocessing increased test accuracy from 0.8806 to 0.9204. The best F-score was obtained with HOG/color features and a validation-selected threshold, reaching 0.9223 on the held-out test set.

For future studies, accuracy could be improved by using a pretrained CNN with transfer learning, segmenting the tongue region before classification, performing a wider hyperparameter search, using k-fold cross-validation, and collecting images under more controlled illumination. A larger dataset would also make augmentation and deeper neural networks more reliable.