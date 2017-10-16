---
title: neural network 损失函数非凸性一例
---

{{ page.title }}
===============

### 微博上[@爱可可](http://weibo.com/p/1005051402400261/home?is_all=1)转发了一个 quora 上关于证明神经网络损失函数[非凸证明的讨论](https://www.quora.com/How-can-you-prove-that-the-loss-functions-in-Deep-Neural-nets-are-non-convex)，回答者是鼎鼎大名的 Ian Goodfellow。我不确定完整理解了 Gooodfellow 的意思，但是以此为启发，结合之前在 cs231n 课程上学到的关于可视化损失函数在低维度上图像的方法，我大致形成了证明这个问题的方法。

### 先描述一下问题，定义 $L(W;X,Y)$ 为某用于分类任务的神经网络的损失函数，其中 $W$ 表示所有的可训练参数，$X, Y$ 分布表示训练数据中的特征和 label。一般来讲 $X,Y$ 是固定的，于是我们即是要证明 $L$ 相对于 $W$ 有可能是非凸的。

### 以下提供一个证明步骤，可能并不那么严谨，但应该足以说明问题：       

1. 构造一个神经网络实例，并将其损失函数用代码实现。
2. 生成两个随机向量 $W_0, W_1$ ，构造高维参数定义域 $W$ 的一个一维子集 $W_0+a{\cdot}W_1$ ，其中 $a$ 是可变标量。
3. 定义函数 $f(a) = L(W_0+a{\cdot}W_1)$ ，证明一元函数 $f$ 是非凸的。
4. 由 $f$ 的非凸性推出 $L$ 的非凸性。


```python
import numpy as np
```


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# %matplotlib inline

%pylab inline
pylab.rcParams['figure.figsize'] = (30, 20)
```

    Populating the interactive namespace from numpy and matplotlib


### 定义非线性激活函数 sigmoid。


```python
def sigmoid(x):
    return 1/(1+np.exp(-x))
```

### 定义一个单隐层的全连接神经网络。我们假定该神经网络用于二分类任务，损失函数使用交叉熵。


```python
def nn_func(x, y):
    def apply(hw, hb, lw, lb):
        p = sigmoid(sigmoid(x.dot(hw)+hb).dot(lw)+lb)
        return -np.mean((y * np.log(p) + (1-y)*np.log(1-p)))
    return apply
```

### 生成一些训练数据，也就是上面提到的 $X,Y$ 。


```python
img, label = np.random.uniform(-0.5, 0.5, size=(10,28*28)), (np.random.uniform(size=(10,1)) > 0.5).astype(np.float32)
```

### 生成上面提到的 $W_0, W_1$ ，根据提到的神经网络定义，参数包含隐层 weight，隐层 bias，输出层 weight，输出层 bias。


```python
base_hw, base_hb, base_lw, base_lb = \
    np.random.uniform(-0.5, 0.5, size=(28*28,100)),\
    np.random.uniform(-0.5, 0.5, size=(1,100)),\
    np.random.uniform(-0.5, 0.5, size=(100,1)),\
    np.random.uniform(-0.5, 0.5, size=(1,1))

dir_hw, dir_hb, dir_lw, dir_lb = \
    np.random.uniform(-0.5, 0.5, size=(28*28,100)),\
    np.random.uniform(-0.5, 0.5, size=(1,100)),\
    np.random.uniform(-0.5, 0.5, size=(100,1)),\
    np.random.uniform(-0.5, 0.5, size=(1,1))
```


```python
loss_func = nn_func(img, label)
loss_func(base_hw, base_hb, base_lw, base_lb)
```




    1.4699588455846047



### 到这里我们已经能够定义 $f$ 函数了，相对于严格的数学证明，以下我采用不那么严谨的可视化方法来进行说明。我们画出 $f$ 在一定范围内的图像，观察其是否非凸。


```python
gma = np.linspace(-1.0,3.0,10000)
loss = []
```


```python
for t in gma:
    hw = base_hw + t*dir_hw
    hb = base_hb + t*dir_hb
    lw = base_lw + t*dir_lw
    lb = base_lb + t*dir_lb
    loss.append(loss_func(hw, hb, lw, lb))
```


```python
plt.plot(np.array(gma), loss)
```




    [<matplotlib.lines.Line2D at 0x12238cfd0>]




![png](https://gameofdimension.github.io/images/output_15_1.png)


### 从上面的图像可以很明显的看出来 $f$ 是非凸的。下面就来从 $f$ 的非凸性推导 $L$ 的非凸性。

由 $f$ 非凸，可知有 $a_1, a_2, \alpha, \beta$ 满足 $f(\alpha\cdot{a_1} + \beta\cdot{a_2}) > \alpha\cdot{f(a_1)} + \beta\cdot{f(a_2)}$，其中 $\alpha > 0, \beta > 0, \alpha + \beta = 1$。    

由 $f$ 与 $L$ 的关系，我们有 $f(\alpha\cdot{a_1} + \beta\cdot{a_2}) = L(W_0+({\alpha\cdot{a_1} + \beta\cdot{a_2}}){\cdot}W_1) > \alpha\cdot{L(W_0+a_1{\cdot}W_1)} + \beta\cdot{L(W_0+a_2{\cdot}W_1)}$     

而 $L(W_0+({\alpha\cdot{a_1} + \beta\cdot{a_2}}){\cdot}W_1) = L(\alpha\cdot(W_0+{a_1}\cdot{W_1}) + \beta\cdot(W_0 + {a_2}\cdot}W_1))$，从而 $L(\alpha\cdot(W_0+{a_1}\cdot{W_1}) + \beta\cdot(W_0 + {a_2}\cdot}W_1)) > \alpha\cdot{L(W_0+a_1{\cdot}W_1)} + \beta\cdot{L(W_0+a_2{\cdot}W_1)}$ ，而这正好说明了 $L$ 的非凸性。

### 到此我们证明了存在有些神经网络损失函数是非凸的。

另外对于由 $f$ 的非凸性推导 $L$ 的非凸性，或许我们可以借助一个低维类比来获得一些直观理解。     

我们假定 $W$ 只有2维，那么整个 $L$ 的图像就可以在三维坐标中展示出来，其形状我们假设是个可能不那么规则的碗状。上面我们做的事情就是在某个位置，从正上方到正下方垂直向这个碗劈一刀，留下的截面就只是一个类似上面的曲线而已。如果我们在这条曲线上发现了一个非凸的实例，那么我们再把视野放到整个碗状图像，这个非凸实例依然是成立的。


```python

```


