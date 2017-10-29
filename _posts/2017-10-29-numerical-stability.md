---
title: 深度学习中的数值稳定性问题一例
---

{{ page.title }}
===============

```python
class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()
```

### 上面[这段代码](https://github.com/pytorch/pytorch/issues/751)中的 forward 函数用来计算所谓的 binary cross entropy，其实也就是逻辑回归中的损失函数
\\[ L(y,x) = -y \cdot log(sigmoid(x))-(1-y) \cdot log(1-sigmoid(x))\\]
### 其中 sigmoid 函数定义如下：
\\[ sigmoid(x) = \frac {1} {1+e^{-x}} \\]

### 既然只是为了计算各交叉熵而已，为什么要搞这么复杂的两行呢：
```python
neg_abs = - input.abs()
loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
```
### 原因在于 \\( e^x \\) 在\\( x \\)是较大正数的时候会出现溢出，也就是超出计算机能表达的范围，从而造成计算误差，而上面这段代码就能在避免计算较大正数的自然指数的同时依然能正确的计算交叉熵。可以发现上面代码的指数运算其操作数是非正数。
### 下面我们检查下这段代码的正确性，代码中的 input 对应 \\(L\\) 定义中的 \\(x\\)，target 对应 \\(L\\) 定义中的 \\(y\\)。
### input.clamp(min=0) 的意思是 \\( max(0,x)\\)，下面分四种情况讨论这个方法的正确性：
1. \\( x \ge 0 \\) 且 \\( y=1 \\)，此时代码计算的是：\\( x-x+log(1+e^{-x}) = -log(\frac {1} {1+e^{-x}})\\)，显然是正确的。
2. \\( x \ge 0 \\) 且 \\( y=0 \\)，此时代码计算的是：\\( x-0+log(1+e^{-x}) = log(1+e^x) = -log(1-\frac {1} {1+e^{-x}})\\)，也是正确的。
3. \\( x \lt 0 \\) 且 \\( y=1 \\)，此时代码计算的是：\\( 0-x+log(1+e^{x}) = log(1+e^{-x})\\)，是正确的。
4. \\( x \lt 0 \\) 且 \\( y=0 \\)，此时代码计算的是：\\( 0-0+log(1+e^{x}) = log(1+e^{-x})= -log(1-\frac {1} {1+e^{-x}})\\)，同样是正确的。

### 在 softmax 的计算中也有指数运算，也存在这样的数值稳定性问题：

\\[ p_k = \frac {e^{x_k}} {\sum e^{x_i}}\\]

### 解决思路也是一样的，就是避免较大正数的自然指数运算：

\\[ p_k = \frac {e^{(x_k-max)}} {\sum e^{(x_i-max)}}\\]

### max 表示所有 \\(x_i\\) 的最大数，这样的一个变化相当于分子分母同除以一个正数，所以计算结果是不变的，另外每个 \\(x_i-max\\) 显然都小于0，于是就不存在大正数的自然指数运算这样的问题了。


