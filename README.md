# Dynamic-Forward-and-Backward-Sparse-Training
Code for the paper 'Dynamic Forward and Backward Sparse Training (DFBST): Accelerated Deep Learning through Completely Sparse Training Schedule' presented at the 14th Asian Conference on Machine Learning (ACML 2022)

Link to paper: https://proceedings.mlr.press/v189/pote23a.html

Steps to follow 
```
cd Dynamic-Forward-and-Backward-Sparse-Training/
```

**Training VGG-16 on CIFAR-10**

```
python train.py --model=vgg16 --mask --alpha=5e-6 --affix=VGG16
```
The other models mentioned in the paper can be trained by suitably changing the arguments in the command.
