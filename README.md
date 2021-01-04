# AdaFair
This repository is an implementation of the paper ["AdaFair: Cumulative Fairness Adaptive Boosting"](https://dl.acm.org/citation.cfm?id=3357974)

This repo contains the source code for AdaFair as well as AdaCost and SMOTEboost. In addition, the employed datasets are also uploaded. 

The repository has been updated. More analytically, the sequence selection of weak learners (\theta parameter) is not performed on the training set rather on a validation set (33% of the given training set (stratified split)).
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
