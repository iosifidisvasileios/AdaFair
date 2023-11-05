# AdaFair: Cumulative Fairness Boosting

## Overview

This GitHub repository presents an extended version of the AdaFair algorithm, initially introduced in the paper titled ["AdaFair: Cumulative Fairness Adaptive Boosting"](https://dl.acm.org/doi/10.1145/3357384.3357974) (CIKM 2019). This extension ["Parity-based cumulative fairness-aware boosting"](https://link.springer.com/article/10.1007/s10115-022-01723-3) (KAIS 2022) incorporates various parity-based fairness notions, enabling a more comprehensive and adaptive approach to fairness in machine learning models.

## Key Features

- **Ensemble Approach:** Introduces an ensemble method for fairness, modifying the data distribution throughout boosting rounds to emphasize misclassified instances of minority groups.

- **Fairness Cost Calculation:** Implements fairness cost, evaluating performance disparities between protected and non-protected groups, ensuring fairness adaptations during model training.

- **Cumulative Notion of Fairness:** Evaluates fairness based on the partial ensemble, providing a cumulative perspective rather than individual boosting rounds.

- **Beneficial for Different Notions:** Demonstrates the effectiveness of cumulative fairness for various parity-based fairness notions, enhancing model fairness across different dimensions.

## Updates

The repository has recently been updated to enhance its functionality and usability:

- **Improved Weak Learner Selection:** The selection of weak learners (\theta parameter) is now performed on a dedicated validation set, optimizing the algorithm's performance.

- **Additional Fairness Notions:** Introduces two new fairness notions, namely "Statistical Parity" (AdaFairSP.py) and "Equal Opportunity" (AdaFairEQOP.py), expanding the algorithm's applicability to different fairness criteria.

## How to Use
To utilize this extended AdaFair algorithm in your machine learning projects, follow these steps:

- pip install adafair

## Preprint
This is an extension of our AdaFair algorithm to incorporate other parity-based fairness notions. We propose an ensemble approach to fairness that alters the data distribution over the boosting rounds “forcing” the model to pay more attention to misclassified instances of the minority. This is done using the so-called fairness cost which assesses performance differences between the protected and non-protected groups. The performance is evaluated based on the partial ensemble rather than on the weak model of each boosting round. We show that this cumulative notion of fairness is beneficiary for different parity-based notions of fairness. Interestingly, the fairness costs also help with the performance on the minority class (if there is imbalance). Imbalance is also directly tackled at post-processing by selecting the partial ensemble with the lowest balanced error.

## Contributions and Issues
Contributions and feedback are welcome. If you encounter any issues or have suggestions for improvement, please feel free to create an issue in the repository or submit a pull request.

**Note:** This repository is actively maintained and updated to ensure the highest standards of fairness and performance in machine learning models. Thank you for considering AdaFair for your fairness-aware machine learning tasks.

See jupyter notebook (run_example.ipynb) on how to train and use the model.