# AI Fake Photo Detector --- Complete Project Summary

> **Current best verified result:** 92.60% test accuracy using
> EfficientNetB0 with RGB input on the CIFAKE held-out test set.\
> **Next target:** 95%+ accuracy, while improving generalisation to
> unseen images.

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Problem Statement and
    Objectives](#2-problem-statement-and-objectives)
3.  [Dataset](#3-dataset)
4.  [Development History](#4-development-history)
5.  [Model Performance Comparison](#5-model-performance-comparison)
6.  [Current Deep Learning Workflow](#6-current-deep-learning-workflow)
7.  [Project Directory Structure](#7-project-directory-structure)
8.  [Technologies and Tools](#8-technologies-and-tools)
9.  [Evaluation Metrics](#9-evaluation-metrics)
10. [Current Findings and
    Limitations](#10-current-findings-and-limitations)
11. [Future Improvements and
    Roadmap](#11-future-improvements-and-roadmap)
12. [Current Project Status](#12-current-project-status)
13. [Conclusion](#13-conclusion)

------------------------------------------------------------------------

## 1. Project Overview

AI Fake Photo Detector is an image-classification project that
investigates Machine Learning, Deep Learning, image feature extraction,
and Transfer Learning techniques to distinguish authentic photographs
from AI-generated synthetic images.

The project began with traditional machine learning and manually
extracted image features. It progressed through advanced feature
engineering, CNN-based feature extraction, and end-to-end deep learning
architectures.

The development progression is:

**Traditional Machine Learning → Advanced Feature Engineering → CNN
Feature Extraction → Transfer Learning → Deep Learning Architecture
Comparison → Colour-Space Experimentation.**

The project contains two major implementation areas:

-   **R:** Dataset preparation, image feature extraction, traditional
    machine learning, Random Forest experiments, and related evaluation.
-   **Python:** CNN-based transfer learning, EfficientNet experiments,
    image preprocessing, training, and testing.

The current best verified result is 92.60% test accuracy using
EfficientNetB0 with RGB input on the CIFAKE held-out test set.

## 2. Problem Statement and Objectives

### 2.1 Problem Statement

AI image-generation technologies can produce highly realistic images.
These images may be difficult to distinguish from genuine photographs
through ordinary visual inspection.

The project investigates computational techniques that analyse image
characteristics and classify an image into one of two categories:

-   **REAL:** An authentic photograph.
-   **FAKE:** An AI-generated image.

### 2.2 Objectives

1.  Prepare and organise a labelled dataset containing real and
    AI-generated images.
2.  Apply image preprocessing techniques suitable for machine learning
    and deep learning.
3.  Extract useful image features using statistical methods and
    CNN-based techniques.
4.  Train and compare traditional machine learning classifiers.
5.  Explore transfer learning using pretrained CNN architectures.
6.  Evaluate models using accuracy, precision, recall, F1-score, and
    confusion matrices.
7.  Investigate alternative image representations, including RGB and
    YCbCr.
8.  Improve classification performance while reducing overfitting.
9.  Evaluate model generalisation on unseen and independent external
    datasets.
10. Develop a reliable image authenticity classification system.

The current performance target is **at least 95% classification
accuracy**. This is an experimental goal, not an achieved or guaranteed
result. Reliability and generalisation are also important.

## 3. Dataset

### 3.1 Dataset

The project uses the **CIFAKE dataset**, containing real images and
AI-generated synthetic images.

### 3.2 Dataset Distribution

  Category      Total Images   Training Images   Test Images
  ----------- -------------- ----------------- -------------
  REAL                 3,000             2,500           500
  FAKE                 3,000             2,500           500
  **Total**        **6,000**         **5,000**     **1,000**

### 3.3 Training, Validation, and Test Split

For Python deep learning experiments, the 5,000 designated training
images are divided using a stratified 80:20 split with random seed 42.

  Dataset portion        Images Purpose
  ----------------- ----------- -------------------------------------------------
  Training                4,000 Used to update model weights
  Validation              1,000 Used to monitor training and select checkpoints
  Held-out test           1,000 Used for evaluation
  **Total**           **6,000** 

The validation split is created from the designated training images. The
1,000 held-out test images are excluded from training and validation.

### 3.4 Dataset Organisation

``` text
data/
├── real/
├── fake/
└── raw/
```

Additional CSV files store datasets and extracted features:

``` text
data/
├── dataset.csv
├── features.csv
├── advanced_features.csv
├── cnn_dataset.csv
└── cnn_features.csv
```

------------------------------------------------------------------------

## 4. Development History

### Stage 1 --- Initial Machine Learning Model

**Approach:** Traditional Machine Learning with manually extracted image
features.

The first implementation used R programming and traditional machine
learning techniques. Images were converted into numerical features
before being supplied to a classifier.

The general process was:

1.  Load REAL and FAKE image datasets.
2.  Prepare and preprocess images.
3.  Extract numerical image characteristics.
4.  Construct a structured dataset.
5.  Train a machine learning classifier.
6.  Evaluate predictions using test data.

The initial feature-based approach used image characteristics such as:

-   RGB colour information.
-   Image brightness.
-   Colour distribution.
-   Statistical image properties.

Extracted information was stored in CSV files, including `features.csv`
and `dataset.csv`.

**Implementation files:**

``` text
R/
├── 01_create_dataset.R
├── 02_prepare_images.R
├── 03_extract_features.R
├── 04_train_model.R
└── 05_test_model.R
```

**Result: 64.80% accuracy.**

This established the initial machine learning baseline.

### Stage 2 --- Random Forest and Advanced Feature Engineering

**Approach:** Random Forest classification with expanded image feature
extraction.

After the initial model achieved 64.80%, the project explored Random
Forest classification and additional feature engineering. The objective
was to provide the classifier with more informative numerical
descriptions of each image.

The feature extraction process was expanded, and the resulting features
were stored in:

``` text
data/
├── features.csv
└── advanced_features.csv
```

**Random Forest improvement:** 75.40% reported accuracy.

Further feature engineering and model experimentation produced the
advanced Random Forest result of **79.90%**.

**Implementation files:**

``` text
R/
├── 03_extract_advanced_features.R
├── 04_train_advanced_random_forest.R
└── 05_test_advanced_random_forest.R
```

**Saved model:**

``` text
models/
└── advanced_random_forest.rds
```

This stage improved on the initial baseline but remained dependent on
manually or statistically extracted image characteristics.

### Stage 3 --- CNN Feature Extraction with Random Forest

**Approach:** CNN-based feature extraction followed by Random Forest
classification.

This stage introduced learned image representations. CNNs can learn
hierarchical visual patterns, including edges, textures, shapes, local
patterns, and higher-level visual representations.

CNN-extracted numerical features were supplied to a Random Forest
classifier. This is a hybrid feature-extraction and classification
approach, rather than an end-to-end CNN classifier.

**Workflow:**

``` text
REAL and FAKE input images
          |
          v
CNN feature extraction
          |
          v
Numerical feature representations
          |
          v
cnn_features.csv / CNN dataset
          |
          v
Random Forest classifier
          |
          v
REAL / FAKE prediction
```

**Implementation files:**

``` text
R/
├── 03_extract_cnn_features.R
├── 04_train_cnn_random_forest.R
└── 05_test_cnn_random_forest.R
```

**Related files:**

``` text
data/
├── cnn_dataset.csv
└── cnn_features.csv

models/
└── cnn_random_forest.rds

results/
└── cnn_test_predictions.csv
```

**Accuracy:** A verified final accuracy figure is not available in the
current project history. The experiment is recorded as implemented, but
its performance should be recovered from the original evaluation output
before reporting a numerical result.

This stage was an important transition from manually designed features
to learned CNN feature representations.

### Stage 4 --- MobileNetV2 Transfer Learning

**Model:** MobileNetV2.

This was the first major end-to-end transfer learning experiment.
Instead of manually extracting features and passing them to a Random
Forest classifier, the project used a pretrained CNN with a custom
binary classification head.

MobileNetV2 is a lightweight convolutional neural network architecture
designed for efficient image feature extraction.

#### Training approach

1.  Load CIFAKE training images.
2.  Resize images to 224 × 224.
3.  Apply MobileNetV2-compatible preprocessing.
4.  Load ImageNet-pretrained MobileNetV2 weights.
5.  Add a custom binary classification head.
6.  Train the classification head.
7.  Fine-tune the network.
8.  Evaluate the model on the 1,000 held-out test images.

#### General architecture

``` text
Input Image (224 × 224 × 3)
          |
          v
MobileNetV2 (ImageNet weights)
          |
          v
Global Average Pooling
          |
          v
Dropout
          |
          v
Dense Layer
          |
          v
Dropout
          |
          v
Sigmoid Output
          |
          v
REAL / FAKE
```

**Implementation files:**

``` text
python/
├── train_mobilenetv2.py
└── test_mobilenetv2.py
```

**Saved models:**

``` text
models/
├── mobilenetv2_best.keras
└── mobilenetv2_final.keras
```

**Saved evaluation and analysis files:**

``` text
results/
├── mobilenetv2_accuracy.png
├── mobilenetv2_loss.png
├── mobilenetv2_test_predictions.csv
├── mobilenetv2_errors.csv
└── mobilenetv2_error_contact_sheet.png
```

**Experimental result: MobileNetV2 test accuracy --- 90.30%.**

  Metric                        Result
  ----------------------- ------------
  Test images                    1,000
  Correct predictions              903
  Incorrect predictions             97
  Test accuracy             **90.30%**

The experiment also included accuracy and loss graphs, confusion matrix,
classification report, incorrect prediction analysis, and an error
contact sheet.

This was a substantial improvement over the earlier traditional machine
learning experiments.

### Stage 5 --- EfficientNetB0 with RGB

**Model:** EfficientNetB0.

After MobileNetV2 achieved 90.30%, EfficientNetB0 was introduced as
another transfer learning architecture to investigate whether
classification performance could be improved.

EfficientNetB0 uses compound scaling and mobile inverted bottleneck
convolution blocks.

#### Implementation details

  Parameter                      Configuration
  ------------------------------ ---------------------
  Architecture                   EfficientNetB0
  Input resolution               224 × 224
  Input channels                 RGB (3 channels)
  Pretrained weights             ImageNet
  Loss function                  Binary Crossentropy
  Optimizer                      Adam
  Initial learning rate          0.001
  Fine-tuning learning rate      0.00001
  Batch size                     16
  Classification output          Sigmoid
  Maximum head-training epochs   12
  Maximum fine-tuning epochs     12

#### Two-phase training

**Phase 1 --- Classification head training**

-   Freeze the EfficientNetB0 backbone.
-   Train the newly added classification head.
-   Use a maximum of 12 epochs.

**Phase 2 --- Fine-tuning**

-   Unfreeze the final 30 layers of EfficientNetB0.
-   Keep BatchNormalization layers frozen.
-   Fine-tune with a smaller learning rate of 0.00001.
-   Use a maximum of 12 epochs.

#### Model architecture

``` text
Input Image (224 × 224 × 3)
          |
          v
EfficientNetB0 (ImageNet weights)
          |
          v
Global Average Pooling
          |
          v
Dropout (0.30)
          |
          v
Dense (128, ReLU)
          |
          v
Dropout (0.20)
          |
          v
Dense (1, Sigmoid)
          |
          v
REAL / FAKE
```

#### Important preprocessing detail

The script decodes JPEG images into three-channel RGB data, resizes them
to 224 × 224, and converts them to `float32`.

The TensorFlow/Keras EfficientNetB0 implementation used in the project
includes an internal rescaling layer. Therefore, the script supplies
pixel values in the 0--255 range and does not apply MobileNetV2's
`preprocess_input`.

**Implementation files:**

``` text
python/
├── train_efficientnetb0.py
└── test_efficientnetb0.py
```

**Saved models:**

``` text
models/
├── efficientnetb0_best.keras
└── efficientnetb0_final.keras
```

**Saved results:**

``` text
results/
├── efficientnetb0_accuracy.png
├── efficientnetb0_loss.png
└── efficientnetb0_test_predictions.csv
```

#### Experimental result

**EfficientNetB0 RGB test accuracy: 92.60%.**

This became the project's best verified model result.

**Confusion matrix** --- rows represent actual labels; columns represent
predicted labels.

  Actual / Predicted     Predicted REAL   Predicted FAKE
  -------------------- ---------------- ----------------
  Actual REAL                       462               38
  Actual FAKE                        36              464

**Classification report:**

  Class            Precision   Recall     F1-score     Support
  -------------- ----------- -------- ------------ -----------
  REAL                0.9277   0.9240       0.9259         500
  FAKE                0.9243   0.9280       0.9261         500
  **Accuracy**                          **0.9260**   **1,000**

EfficientNetB0 RGB improved on the MobileNetV2 result by 2.30 percentage
points and became the strongest model in the project.

### Stage 6 --- EfficientNetB0 with YCbCr

**Model:** EfficientNetB0 with YCbCr input.

After achieving 92.60% using RGB, the project investigated whether
changing the image colour representation could improve classification
performance.

The experiment converted RGB images into YCbCr before supplying them to
EfficientNetB0.

#### What is YCbCr?

YCbCr separates an image into three components:

  Channel   Description
  --------- -----------------------------
  Y         Luminance or brightness
  Cb        Blue-difference chrominance
  Cr        Red-difference chrominance

The intention was to investigate whether separating luminance and
chrominance information could provide a useful representation for
distinguishing real photographs from AI-generated images.

#### RGB to YCbCr conversion

The training and test scripts use the following conversion:

``` python
red = image[..., 0]
green = image[..., 1]
blue = image[..., 2]

y = (
    0.299000 * red
    + 0.587000 * green
    + 0.114000 * blue
)

cb = (
    128.0
    - 0.168736 * red
    - 0.331264 * green
    + 0.500000 * blue
)

cr = (
    128.0
    + 0.500000 * red
    - 0.418688 * green
    - 0.081312 * blue
)

ycbcr = tf.stack([y, cb, cr], axis=-1)
```

The output channel order is Y, Cb, Cr. The values are clipped to the
0--255 range.

#### Training configuration

  Parameter                      Configuration
  ------------------------------ ---------------------
  Architecture                   EfficientNetB0
  Input representation           YCbCr
  Input channels                 3
  Input resolution               224 × 224
  Pretrained weights             ImageNet
  Loss function                  Binary Crossentropy
  Optimizer                      Adam
  Initial learning rate          0.001
  Fine-tuning learning rate      0.00001
  Batch size                     16
  Maximum head-training epochs   12
  Maximum fine-tuning epochs     12

#### Important difference from the RGB experiment

The RGB training script fine-tunes only the final 30 layers of the
EfficientNetB0 backbone.

The YCbCr experiment unfreezes almost the entire backbone, while keeping
BatchNormalization layers frozen.

This was intended to allow the pretrained convolutional filters to adapt
to the changed input representation. As a result, the YCbCr experiment
requires more computation and takes longer to train.

#### Implementation files

``` text
python/
├── train_efficientnetb0_ycbcr.py
└── test_efficientnetb0_ycbcr.py
```

#### Saved models

``` text
models/
├── efficientnetb0_ycbcr_best.keras
└── efficientnetb0_ycbcr_final.keras
```

#### Saved results

``` text
results/
├── efficientnetb0_ycbcr_accuracy.png
├── efficientnetb0_ycbcr_loss.png
└── efficientnetb0_ycbcr_test_predictions.csv
```

#### Training observations

The training output showed:

-   Training accuracy reached 100% in the reported late epoch.
-   Validation accuracy reached 92.20% at the reported epoch.
-   Training loss was 0.0112 in the final displayed epoch.
-   Validation loss was 0.2575 in the final displayed epoch.
-   EarlyStopping terminated the training before the maximum epoch
    count.
-   The callback reported restoring weights from the best
    validation-loss epoch, epoch 4.

These observations indicate a training/validation generalisation gap and
possible overfitting.

**Important callback distinction:** The best-checkpoint file is saved
according to validation accuracy, whereas EarlyStopping monitors
validation loss. Consequently, the best-accuracy checkpoint and the
final model saved after restoring the best-loss weights may contain
different weights.

#### Experimental test result

**EfficientNetB0 YCbCr test accuracy: 92.60%.**

  Metric                        Result
  ----------------------- ------------
  Test images                    1,000
  Correct predictions              926
  Incorrect predictions             74
  Test accuracy             **92.60%**

**Confusion matrix:**

  Actual / Predicted     Predicted REAL   Predicted FAKE
  -------------------- ---------------- ----------------
  Actual REAL                       469               31
  Actual FAKE                        43              457

**Classification report:**

  Class            Precision   Recall     F1-score     Support
  -------------- ----------- -------- ------------ -----------
  REAL                0.9160   0.9380       0.9269         500
  FAKE                0.9365   0.9140       0.9251         500
  **Accuracy**                          **0.9260**   **1,000**

#### Comparison with RGB EfficientNetB0

  Metric                         RGB    YCbCr
  ------------------------- -------- --------
  REAL correctly detected        462      469
  FAKE correctly detected        464      457
  REAL misclassified              38       31
  FAKE misclassified              36       43
  Correct predictions            926      926
  Incorrect predictions           74       74
  Test accuracy               92.60%   92.60%

#### Conclusion

The YCbCr experiment did not improve overall test accuracy over the RGB
EfficientNetB0 baseline.

However, it changed the distribution of classification errors:

-   YCbCr correctly classified seven more REAL images.
-   RGB correctly classified seven more FAKE images.
-   Both models produced 926 correct predictions and 74 incorrect
    predictions.

The experiment demonstrates that changing the colour representation
affects model predictions, but it has not demonstrated an overall
accuracy improvement.

------------------------------------------------------------------------

## 5. Model Performance Comparison

### 5.1 Accuracy Table

  Model / Experiment                             Accuracy Status
  ---------------------------------------- -------------- ----------------------
  Initial Machine Learning                         64.80% Completed
  Random Forest                                    75.40% Completed
  Advanced Random Forest                           79.90% Completed
  CNN Feature Extraction + Random Forest     Not verified Implemented
  MobileNetV2                                      90.30% Completed
  EfficientNetB0 RGB                           **92.60%** Best verified result
  EfficientNetB0 YCbCr                             92.60% Completed

### 5.2 Accuracy Progression

``` text
Initial Machine Learning       64.80%
Random Forest                   75.40%
Advanced Random Forest          79.90%
MobileNetV2                     90.30%
EfficientNetB0 RGB              92.60%
EfficientNetB0 YCbCr            92.60%
```

The improvement from the initial model to the current best model is:

``` text
92.60 - 64.80 = 27.80 percentage points
```

The project has improved by **27.80 percentage points** from its initial
reported result.

### 5.3 Current Best Model

**EfficientNetB0 with RGB input --- 92.60% test accuracy.**

This is the current best verified result among the reported experiments.
The YCbCr experiment achieved the same overall accuracy, so it does not
replace RGB as a superior model based on accuracy.

------------------------------------------------------------------------

## 6. Current Deep Learning Workflow

The current Python deep learning workflow follows these stages:

``` text
                 CIFAKE DATASET
                REAL and FAKE images
                         |
                         v
                IMAGE PREPROCESSING
             Decode, resize, and convert
                         |
                         v
              TRAINING / VALIDATION
                 Stratified 80:20
                         |
                         v
                 CNN MODEL TRAINING
              Transfer learning and
                   fine-tuning
                         |
                         v
                BEST MODEL CHECKPOINT
              Validation-based selection
                         |
                         v
                HELD-OUT TEST EVALUATION
                         |
                         v
                 PERFORMANCE ANALYSIS
           Accuracy, precision, recall, F1
                 and confusion matrix
                         |
                         v
                PREDICTION ANALYSIS
              CSV and error inspection
```

### 6.1 Two-Phase Training

The MobileNetV2 and EfficientNetB0 experiments use transfer-learning
strategies with a classification-head training stage followed by
fine-tuning.

**Phase 1 --- Classification head training**

-   Load the ImageNet-pretrained backbone.
-   Freeze the backbone.
-   Train the newly added classification layers.
-   Monitor validation accuracy and validation loss.

**Phase 2 --- Fine-tuning**

-   Unfreeze selected backbone layers or, in the YCbCr experiment,
    almost the entire backbone.
-   Keep BatchNormalization layers frozen.
-   Use a smaller learning rate.
-   Continue training to adapt pretrained features to the project
    dataset.

### 6.2 Training Callbacks

The EfficientNetB0 scripts use:

**ModelCheckpoint:** Saves the best model checkpoint according to
validation accuracy.

**EarlyStopping:** Monitors validation loss and stops training when it
has not improved for the configured patience period. It can restore the
best validation-loss weights.

**ReduceLROnPlateau:** Reduces the learning rate when validation loss
stops improving.

### 6.3 Epochs and Batches

The EfficientNetB0 experiments use:

-   Batch size: 16.
-   Head-training maximum: 12 epochs.
-   Fine-tuning maximum: 12 epochs.

An epoch represents one complete pass through the training dataset. With
4,000 training images and batch size 16, there are 250 batches per
epoch.

The configured maximum is 24 epochs across the two phases, but callbacks
may stop either phase earlier.

### 6.4 Matching Training and Test Preprocessing

The test pipeline must use the same image decoding, resizing, colour
conversion, and pixel-range conventions as the corresponding training
pipeline.

For example:

-   RGB EfficientNetB0 must be tested using the RGB preprocessing used
    during its training.
-   YCbCr EfficientNetB0 must be tested using the same RGB-to-YCbCr
    conversion used during its training.

Passing RGB images directly to the YCbCr model would produce an invalid
comparison.

------------------------------------------------------------------------

## 7. Project Directory Structure

The following is the project structure recorded during the latest
development update:

``` text
AI_Fake_Photo_Detector/
│
├── -
├── app/
│
├── data/
│   ├── advanced_features.csv
│   ├── cnn_dataset.csv
│   ├── cnn_features.csv
│   ├── dataset.csv
│   ├── fake/
│   ├── features.csv
│   ├── raw/
│   └── real/
│
├── LICENSE
│
├── models/
│   ├── advanced_random_forest.rds
│   ├── cnn_random_forest.rds
│   ├── efficientnetb0_best.keras
│   ├── efficientnetb0_final.keras
│   ├── efficientnetb0_ycbcr_best.keras
│   ├── efficientnetb0_ycbcr_final.keras
│   ├── fake_photo_model.rds
│   ├── fake_photo_random_forest.rds
│   ├── mobilenetv2_best.keras
│   └── mobilenetv2_final.keras
│
├── p
├── plots/
│
├── python/
│   ├── inspect_errors.py
│   ├── test_efficientnetb0.py
│   ├── test_efficientnetb0_ycbcr.py
│   ├── test_mobilenetv2.py
│   ├── train_efficientnetb0.py
│   ├── train_efficientnetb0_ycbcr.py
│   └── train_mobilenetv2.py
│
├── R/
│   ├── 01_create_dataset.R
│   ├── 02_prepare_images.R
│   ├── 03_extract_advanced_features.R
│   ├── 03_extract_cnn_features.R
│   ├── 03_extract_features.R
│   ├── 04_train_advanced_random_forest.R
│   ├── 04_train_cnn_random_forest.R
│   ├── 04_train_model.R
│   ├── 04_train_random_forest.R
│   ├── 05_test_advanced_random_forest.R
│   ├── 05_test_cnn_random_forest.R
│   ├── 05_test_model.R
│   ├── 05_test_random_forest.R
│   ├── 06_train_mobilenetv2.R
│   └── test.r
│
├── README.md
│
└── results/
    ├── advanced_random_forest_predictions.csv
    ├── cnn_test_predictions.csv
    ├── efficientnetb0_accuracy.png
    ├── efficientnetb0_loss.png
    ├── efficientnetb0_test_predictions.csv
    ├── efficientnetb0_ycbcr_accuracy.png
    ├── efficientnetb0_ycbcr_loss.png
    ├── efficientnetb0_ycbcr_test_predictions.csv
    ├── mobilenetv2_accuracy.png
    ├── mobilenetv2_error_contact_sheet.png
    ├── mobilenetv2_errors.csv
    ├── mobilenetv2_loss.png
    ├── mobilenetv2_test_predictions.csv
    ├── random_forest_predictions.csv
    └── test_predictions.csv
```

The root entries named `-` and `p` were present in the supplied
directory listing. Their purpose has not been established and should be
checked before repository cleanup.

### 7.1 Main Directory Responsibilities

  -----------------------------------------------------------------------
  Directory                           Responsibility
  ----------------------------------- -----------------------------------
  `app/`                              Application-related code and
                                      integration

  `data/`                             Original images, prepared data, and
                                      extracted feature datasets

  `models/`                           Trained R and Keras model files

  `python/`                           Python training, testing, and
                                      analysis scripts

  `R/`                                R-based data preparation, feature
                                      extraction, training, and testing

  `plots/`                            Project visualisations

  `results/`                          Evaluation outputs, predictions,
                                      graphs, and error analysis
  -----------------------------------------------------------------------

### 7.2 Model File Formats

-   `.rds`: R model objects and saved R data.
-   `.keras`: TensorFlow/Keras model files.
-   `.csv`: Structured datasets, extracted features, and prediction
    results.
-   `.png`: Training plots and error-analysis visualisations.

------------------------------------------------------------------------

## 8. Technologies and Tools

  -----------------------------------------------------------------------
  Category                            Technologies
  ----------------------------------- -----------------------------------
  Programming languages               R, Python

  Traditional Machine Learning        Random Forest

  Deep Learning Framework             TensorFlow, Keras

  CNN Architectures                   MobileNetV2, EfficientNetB0

  Feature Extraction                  Statistical image features, CNN
                                      features

  Image Processing                    Image resizing, RGB, YCbCr

  Data Processing                     NumPy, Pandas

  Model Evaluation                    Scikit-learn

  Visualisation                       Matplotlib

  Dataset                             CIFAKE

  Model formats                       RDS, Keras

  Development environment             Linux, Python virtual environment,
                                      R environment
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 9. Evaluation Metrics

The project uses multiple metrics to assess classification performance.

### 9.1 Accuracy

Accuracy measures the proportion of correctly classified images.

``` text
Accuracy = Correct Predictions / Total Predictions
```

### 9.2 Precision

Precision measures how many images predicted as a particular class
actually belong to that class.

### 9.3 Recall

Recall measures how many images belonging to a class were correctly
identified.

### 9.4 F1-score

The F1-score is the harmonic mean of precision and recall.

### 9.5 Confusion Matrix

A confusion matrix shows the distribution of correct and incorrect
predictions for each class.

For this project:

-   True REAL: REAL image correctly predicted as REAL.
-   REAL-to-FAKE error: REAL image incorrectly predicted as FAKE.
-   FAKE-to-REAL error: FAKE image incorrectly predicted as REAL.
-   True FAKE: FAKE image correctly predicted as FAKE.

The displayed confusion matrices use rows for actual classes and columns
for predicted classes.

### 9.6 Classification Report

The classification report provides precision, recall, F1-score, and
support for both REAL and FAKE classes.

------------------------------------------------------------------------

## 10. Current Findings and Limitations

### 10.1 Best Model

EfficientNetB0 with RGB input is currently the best verified model, with
92.60% test accuracy.

### 10.2 YCbCr Did Not Improve Overall Accuracy

The YCbCr experiment achieved the same 92.60% test accuracy as RGB
EfficientNetB0.

The confusion matrices show that the two models make different
classification errors, even though their total correct predictions are
equal.

### 10.3 Possible Overfitting

The YCbCr experiment showed 100% training accuracy in the reported late
epoch, while validation accuracy was 92.20%.

This indicates a generalisation gap and possible overfitting.

Training accuracy alone should not be treated as evidence of model
quality. Validation and held-out test performance are more informative.

### 10.4 EarlyStopping and Checkpoint Behaviour

The YCbCr training stopped before completing all configured epochs
because EarlyStopping was enabled.

The callback reported restoring weights from epoch 4, the best epoch
according to validation loss.

There is an important distinction between the saved files:

-   `efficientnetb0_ycbcr_best.keras` is saved according to the best
    validation accuracy.
-   `efficientnetb0_ycbcr_final.keras` is saved after the training
    process restores the best validation-loss weights.

These files may therefore contain different model weights.

### 10.5 CPU Training

The development environment previously reported:

-   Intel Core i5-6300U.
-   16 GB RAM.
-   Intel HD Graphics 520.
-   No available NVIDIA CUDA GPU.

TensorFlow reported that CUDA drivers were unavailable and that the GPU
would not be used. Training therefore ran primarily on the CPU.

The YCbCr experiment took longer partly because it performs RGB-to-YCbCr
calculations during preprocessing and unfreezes substantially more of
the EfficientNetB0 backbone during fine-tuning.

The CUDA warning is not itself an indication of an accuracy failure.

### 10.6 Test Dataset Reuse

The same 1,000 CIFAKE held-out test images have been used to compare
multiple experiments.

These results are useful as a development benchmark, but repeated model
selection based on the same test set can gradually bias decisions
towards that particular test set.

A separate external dataset should be used for a stronger final
assessment of generalisation.

### 10.7 Generalisation Limitation

The current results are based on CIFAKE. High accuracy on CIFAKE does
not automatically imply similar accuracy on arbitrary real-world
photographs or images generated by different AI systems.

------------------------------------------------------------------------

## 11. Future Improvements and Roadmap

  -----------------------------------------------------------------------
  Priority                Improvement             Purpose
  ----------------------- ----------------------- -----------------------
  1                       EfficientNetV2B0        Experiment with a
                                                  different EfficientNet
                                                  architecture

  2                       ConvNeXt-Tiny           Explore another modern
                                                  CNN architecture

  3                       Misclassification       Investigate images
                          analysis                current models classify
                                                  incorrectly

  4                       Ensemble learning       Combine complementary
                                                  predictions from strong
                                                  models

  5                       External dataset        Test generalisation
                          evaluation              beyond CIFAKE

  6                       Application integration Develop a usable image
                                                  classification
                                                  interface
  -----------------------------------------------------------------------

### 11.1 EfficientNetV2B0

The next proposed architecture experiment is EfficientNetV2B0.

The aim is to determine whether a different EfficientNet architecture
and feature extraction design can improve classification performance.

This is a planned experiment. No accuracy result is claimed yet.

### 11.2 ConvNeXt-Tiny

ConvNeXt-Tiny is another proposed CNN architecture for image
classification. It can be evaluated as an alternative to MobileNetV2 and
EfficientNetB0.

### 11.3 Misclassification Analysis

The project should investigate the images that existing models classify
incorrectly.

The current EfficientNetB0 RGB result contains 74 incorrect predictions
out of 1,000 test images. The YCbCr model also has 74 incorrect
predictions, although the individual errors differ.

A detailed analysis may help identify:

-   Difficult image categories.
-   Similarities between real and generated images.
-   Image quality issues.
-   Dataset-specific patterns.
-   Possible duplicate or near-duplicate images.
-   Cases where the model relies on dataset-specific shortcuts.

### 11.4 Ensemble Learning

An ensemble experiment could combine predictions from independently
trained models, such as:

-   EfficientNetB0 RGB.
-   MobileNetV2.
-   Other future high-performing models.

The ensemble should be evaluated using validation data before its final
evaluation. Combining models does not guarantee an improvement.

### 11.5 External Dataset Evaluation

Future evaluation should include an independent dataset containing:

-   Authentic photographs from sources outside the training dataset.
-   AI-generated images from different generation systems.
-   Images with different resolutions and compression levels.
-   Images that have been resized, cropped, or compressed.

### 11.6 Application Integration

The project includes an `app/` directory, but its complete
implementation status is not established by the current development
summary.

A future application can allow a user to upload an image and receive:

-   Predicted class: REAL or FAKE.
-   Model confidence or class probability.
-   Selected model information.
-   A clear explanation that the output is a model prediction, not
    definitive proof of image authenticity.

------------------------------------------------------------------------

## 12. Development Roadmap

### 12.1 Completed Milestones

  -----------------------------------------------------------------------
  Milestone               Description             Status
  ----------------------- ----------------------- -----------------------
  Dataset preparation     CIFAKE dataset          Completed
                          organisation and        
                          preparation             

  Initial ML              Initial feature-based   Completed
                          classifier              

  Random Forest           Traditional machine     Completed
                          learning improvement    

  Advanced Random Forest  Expanded feature        Completed
                          engineering             

  CNN feature extraction  CNN features with       Implemented
                          Random Forest           
                          classification          

  MobileNetV2             Transfer learning and   Completed
                          fine-tuning             

  EfficientNetB0 RGB      RGB-based transfer      Completed
                          learning                

  EfficientNetB0 YCbCr    Alternative             Completed
                          colour-space experiment 
  -----------------------------------------------------------------------

### 12.2 Upcoming Milestones

1.  Implement and train EfficientNetV2B0.
2.  Evaluate EfficientNetV2B0 on validation data and the held-out
    benchmark.
3.  Experiment with ConvNeXt-Tiny.
4.  Analyse misclassified images in detail.
5.  Investigate model overfitting and generalisation.
6.  Experiment with ensemble learning.
7.  Evaluate the selected model on an independent external dataset.
8.  Select and document the final model.
9.  Complete application integration and testing.
10. Prepare final project documentation and results.

### 12.3 Performance Goal

The next performance goal is to investigate whether the project can
exceed 95% accuracy.

The final model should be selected based on more than accuracy alone.
Precision, recall, F1-score, error distribution, and external
generalisation must also be considered.

------------------------------------------------------------------------

## 13. Current Project Status

### 13.1 Current Best Model

  Item                    Current result
  ----------------------- ----------------
  Model                   EfficientNetB0
  Input representation    RGB
  Input resolution        224 × 224
  Dataset                 CIFAKE
  Test images             1,000
  Correct predictions     926
  Incorrect predictions   74
  Test accuracy           **92.60%**

### 13.2 Completed Model Files

``` text
models/
├── advanced_random_forest.rds
├── cnn_random_forest.rds
├── efficientnetb0_best.keras
├── efficientnetb0_final.keras
├── efficientnetb0_ycbcr_best.keras
├── efficientnetb0_ycbcr_final.keras
├── fake_photo_model.rds
├── fake_photo_random_forest.rds
├── mobilenetv2_best.keras
└── mobilenetv2_final.keras
```

### 13.3 Current Python Scripts

``` text
python/
├── inspect_errors.py
├── test_efficientnetb0.py
├── test_efficientnetb0_ycbcr.py
├── test_mobilenetv2.py
├── train_efficientnetb0.py
├── train_efficientnetb0_ycbcr.py
└── train_mobilenetv2.py
```

### 13.4 Immediate Next Step

The proposed next experiment is EfficientNetV2B0, followed by detailed
error analysis and independent external dataset evaluation.

------------------------------------------------------------------------

## 14. Conclusion

The AI Fake Photo Detector project has evolved from a traditional
feature-based machine learning implementation into a deep learning-based
image classification research project.

The development began with manually extracted image features and Random
Forest classification. It then progressed to advanced feature
engineering, CNN-based feature extraction, and transfer learning using
MobileNetV2 and EfficientNetB0.

The MobileNetV2 model achieved 90.30% test accuracy. EfficientNetB0 with
RGB input subsequently improved the result to 92.60%, becoming the
strongest verified model in the project.

An additional experiment using YCbCr colour-space conversion was
conducted to investigate whether separating luminance and chrominance
information could improve classification. Although the YCbCr experiment
achieved the same 92.60% test accuracy, it produced a different
distribution of classification errors.

The project has demonstrated a progression across multiple machine
learning and deep learning approaches. The next development phase will
focus on testing additional architectures, reducing overfitting,
analysing misclassified images, exploring ensemble learning, and
evaluating generalisation using independent external datasets.

The long-term objective is to develop a reliable AI-generated image
detection system that performs consistently on unseen images, with a
target of at least 95% classification accuracy.

------------------------------------------------------------------------

## Accuracy and Reporting Notes

-   Reported accuracy values should be tied to the corresponding
    experiment and evaluation split.
-   The CNN Feature Extraction + Random Forest accuracy is intentionally
    marked as unverified because a numerical result was not available in
    the supplied project history.
-   The 92.60% RGB and YCbCr results are results on the same CIFAKE
    held-out test set, not independent external evaluations.
-   The 95% figure is a target, not an achieved result.
-   Model confidence is not the same as verified probability of
    real-world authenticity.
-   Before final publication, confirm the exact versions of libraries,
    dataset provenance, training logs, and evaluation scripts used for
    each experiment.
