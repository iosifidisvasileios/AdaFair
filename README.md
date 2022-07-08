# AdaFair

This is an extension of our AdaFair algorithm (CIKM 2019, ["AdaFair: Cumulative Fairness Adaptive Boosting"](https://dl.acm.org/citation.cfm?id=3357974)) to other parity-based fairness notions. We propose an ensemble approach to fairness that alters the data distribution over the boosting rounds “forcing” the model to pay more attention to misclassified instances of the minority. This is done using the so-called fairness cost which assesses performance differences between the protected and non-protected groups. The performance is evaluated based on the partial ensemble rather than on the weak model of each boosting round. We show that this cumulative notion of fairness is beneficiary for different parity-based notions of fairness. Interestingly, the fairness costs also help with the performance on the minority class (if there is imbalance). Imbalance is also directly tackled at post-processing by selecting the partial ensemble with the lowest balanced error.
Preprint available at : https://lnkd.in/eARuqFDe

The repository has been updated. More analytically, the sequence selection of weak learners (\theta parameter) is not performed on the training set rather on a validation set.
In addition, two fairness notions, namely "statistical parity" and "equal opportunity" have been employed by AdaFair (AdaFairSP.py and AdaFairEQOP.py, respectively).

If you use this paper, please do not forget to cite it:

@inproceedings{iosifidis2019adafair,
  title={AdaFair: Cumulative Fairness Adaptive Boosting},
  author={Iosifidis, Vasileios and Ntoutsi, Eirini},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={781--790},
  year={2019},
  organization={ACM}
}
