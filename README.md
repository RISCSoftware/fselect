# fselect
Repository for feature selection methods.

## RSR

[rsr.py](fselect/rsr/rsr.py)

Unsupervised feature selection based on:

* Zhu, P., Zuo, W., Zhang, L., Hu, Q., & Shiu, S. C. (2015). Unsupervised feature selection by regularized self-representation. Pattern Recognition, 48(2), 438-446.

Learn a matrix W to minimize $$\| X - X W \|_{2,1} + \lambda \|W\|_{2,1}.$$
