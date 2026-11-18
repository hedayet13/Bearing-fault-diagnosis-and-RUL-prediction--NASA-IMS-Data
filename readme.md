\# Bearing Fault Diagnosis and RUL Prediction Using NASA IMS Data



MATLAB implementation and experimental outputs for the study:



\*\*Frequency-domain-based Bayesian-optimized 1-D CNN feature learning for integrated bearing fault diagnosis and remaining useful life prediction\*\*



\## Overview



This project investigates bearing fault diagnosis, degradation assessment, and remaining useful life (RUL) prediction using the publicly available IMS bearing run-to-failure datasets.



The framework combines frequency-domain signal processing, automatic CNN feature learning, Bayesian hyperparameter optimization, support vector machine (SVM) classification, and Gaussian process regression (GPR).



\## Methodology



1\. Transform vibration recordings into frequency-domain representations using the Fast Fourier Transform (FFT).

2\. Train a 1-D CNN using early-stage and late-stage recordings.

3\. Apply Bayesian optimization to select the learning rate, mini-batch size, L-2 regularization coefficient, feature dimension, and convolutional filter counts.

4\. Extract learned features for Gaussian-kernel SVM fault classification.

5\. Combine features at the bearing level to construct health indicators.

6\. Apply GPR to model degradation and estimate RUL through threshold crossing.

7\. Compare the proposed representation with a CNN–LSTM benchmark using a common downstream prognostic procedure.



\## Repository Contents



| Folder | Contents |

| --- | --- |

| `New\_CNN\_based\_features\_1st\_Bearing` | CNN feature learning, SVM diagnosis, and saved results for Dataset 1 |

| `New\_CNN\_based\_features\_2nd\_Bearing` | CNN feature learning, SVM diagnosis, and saved results for Dataset 2 |

| `New\_CNN\_based\_features\_3rd\_Bearing` | CNN feature learning, SVM diagnosis, and saved results for Dataset 3 |

| `CNN\_vs\_CNN-LSTM\_RUL\_1st\_data` | RUL comparison scripts and outputs for Dataset 1 |

| `CNN\_vs\_CNN-LSTM\_RUL\_2nd\_data` | RUL comparison scripts and outputs for Dataset 2 |

| `CNN\_vs\_CNN-LSTM\_RUL\_3rd\_data` | RUL comparison scripts and outputs for Dataset 3 |



Saved outputs include classification reports, confusion matrices, performance summaries, and bearing-wise RUL comparison figures.



\## Dataset



The IMS bearing datasets were provided by the Center for Intelligent Maintenance Systems at the University of Cincinnati and are distributed through NASA’s Prognostics Data Repository.



Dataset access:

https://data.nasa.gov/dataset/ims-bearings



The original recordings and prepared input MAT files are not included in this repository.



The feature-learning scripts expect a MATLAB variable named `Runs`, organized as a cell array in chronological order. Each cell contains one recording as a numerical matrix with signal samples in rows and sensor channels in columns.



\## Requirements



\- MATLAB

\- Deep Learning Toolbox

\- Statistics and Machine Learning Toolbox

\- Parallel Computing Toolbox if GPU execution or parallel processing is enabled



The exact MATLAB release and toolbox versions used for reproduction should be recorded with each experiment.



\## Running the Analysis



1\. Download the IMS recordings and prepare the corresponding `Runs` MAT files.

2\. Open the folder for the desired dataset in MATLAB.

3\. Check the input filenames and local paths in the scripts.

4\. Run the corresponding feature-learning script:

&#x20;  - Dataset 1: `Bayes\_CNN\_feature\_01.m`

&#x20;  - Dataset 2: `Bayes\_CNN\_feature\_02.m`

&#x20;  - Dataset 3: `Bayes\_CNN\_feature\_03.m`

5\. Run the matching SVM classification script:

&#x20;  - Dataset 1: `CNN\_feature\_SVM\_class\_01.m`

&#x20;  - Dataset 2: `CNN\_feature\_SVM\_class\_02.m`

&#x20;  - Dataset 3: `CNN\_feature\_SVM\_class\_03.m`

6\. For RUL comparisons, make both the proposed CNN and benchmark CNN–LSTM feature files available, then run the corresponding `RUL\_dataset\_\*.m` script.



Use each dataset folder separately so MATLAB resolves the intended versions of helper functions with identical names.



\## Current Setup Notes



\- The RUL comparison scripts require benchmark CNN–LSTM feature MAT files that are not currently included.

\- The third-dataset feature script currently loads `IMS\_4th\_test\_Runs.mat` but saves features with an `IMS\_3rd\_test` filename.

\- The third-dataset RUL script currently expects feature filenames containing `IMS\_4th\_test`. Confirm the dataset identity and make these filenames consistent before running the comparison.



\## Evaluation Scope



Models are optimized separately for each dataset. The reported experiments therefore do not establish transfer of a single trained model to an unseen dataset.



CNN optimization uses early-stage versus late-stage classification accuracy rather than RUL prediction error. The prognostic results should be interpreted within the retrospective run-to-failure protocol described in the manuscript.



\## Questions



Please open a GitHub issue for questions about the implementation, dataset preparation, or experimental outputs.

