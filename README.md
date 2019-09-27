# AdaFair
This repository is an implementation of the paper "AdaFair: Cumulative Fairness Adaptive Boosting"

# Abstract:
The widespread use of ML-based decision making in domains with high societal impact such as recidivism, job hiring and loan credit has raised a lot of concerns regarding potential discrimination. In particular, in certain cases it has been observed that ML algorithms can provide different decisions based on sensitive attributes such as gender or race and therefore can lead to discrimination. Although, several fairness-aware ML approaches have been proposed, their focus has been largely on preserving the overall classification accuracy while improving fairness in predictions for both protected and non-protected groups (defined based on the sensitive attribute(s)). The overall accuracy however is not a good indicator of performance in
case of class imbalance, as it is biased towards the majority class. As we will see in our experiments, many of the fairness-related datasets suffer from class imbalance and therefore, tackling fairness requires also tackling the imbalance problem.
To this end, we propose AdaFair , a fairness-aware classifier based on AdaBoost that further updates the weights of the instances in each boosting round taking into account a cumulative notion of fairness based upon all current ensemble members, while explicitly tackling class-imbalance by optimizing the number of ensemble members for balanced classification error. Our experiments show that our approach can achieve parity in true positive and true negative rates for both protected and non-protected groups, while it significantly outperforms existing fairness-aware methods up to 25% in terms of balanced error.

This repo contains the source code for AdaFair as well as AdaCost and SMOTEboost. In addition, the employed datasets are also uploaded. 

If you use this paper please do not forget to cite it:

@article{iosifidis2019adafair,
  title={AdaFair: Cumulative Fairness Adaptive Boosting},
  author={Iosifidis, Vasileios and Ntoutsi, Eirini},
  journal={arXiv preprint arXiv:1909.08982},
  year={2019}
}
