# RNN 代码

## 1 从零开始

### 1 引入依赖

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from matplotlib import pyplot as plt
```

### 2 设置超参数，获取训练数据

```python
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

### 3 one_hot 编码初体验

```python
"""
one-hot 编码
"""
one_hot_0_2 = F.one_hot(torch.tensor([0, 2]), len(vocab))  # 索引为 0 和 2 的 one_hot 向量
# print(one_hot_0_2)
# tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0]])
# len(vocab) == 28

"""
X = torch.arange(15) 生成列表 [0, 14] 
reshape((3, 5)) 将列表维度改成 3行 5列
F.one_hot(X.T) 生成 5 个 3行 15列 的子单词序列
5 个对应 X.T 5 行
3 行 对应 X.T 3 列
15 列 对应 len(X.T) == 15
默认按照第 0 维划分，也就是按照每一行进行划分
将每一行作为一个子序列，共有 行数个 子序列， 每个子序列中变量个数为 列数个，每一个变量的列为 len(vocab) 个
"""
X = torch.arange(10).reshape((2, 5))


# print(F.one_hot(X))
# print(F.one_hot(X.T, len(vocab)).shape)
# 注意第二个维度不能省略，如果省略，则生成张量中每个子张量的长度为 len(X.T)
# torch.Size([5, 2, 28])
# num_steps 5
# for x in X.T: x shape (2, 28) (batch_size, len(X.T))
```

### 4 绘图部分

```python
"""
绘图函数和类
"""
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.legend = legend
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xscale = xscale
    plt.yscale = yscale
    if legend:
        axes.legend(legend)
    axes.grid()

class MyPlotMethod:
    def __init__(self,  xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        plt.rcParams['figure.figsize'] = figsize
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
```

### 5 RNN模型部分

#### 1 生成模型参数

```python
def get_params(vocab_size, num_hiddens, device):
    """
    生成模型计算需要的各种参数

    模型流程
    1. 计算当前层的初始隐藏状态：H_o = X * W_xh + H * W_hh + b_h
    2. 将初始隐藏状态经过激活函数转化成为真正的当前隐藏状态：H = 激活函数(H_o)
    3. 计算输出：Y = H * W_hq + b_q

    最后计算出的 H shape (batch_size, num_hiddens)
    Y shape (batch_size, vocab_size)

    X: inputs shape (batch_size, vocab_size)
    W_xh: inputs 的权重 shape (vocab_size, num_hiddens)
    H: hidden units shape (batch_size, num_hiddens)
    W_hh: 隐藏状态的权重 shape (num_hiddens, num_hiddens)
    b_h: 计算当前隐藏状态需要的偏差 shape num_hiddens
    new_H:  shape (batch_size, num_hiddens)
    W_hq: 输出层权重 shape(num_hiddens, vocab_size)
    b_q: 输出层偏差 shape vocab_size
    Y: outputs (batch_size, vocab_size) 形状又变成和 inputs 相同的形状
    :param vocab_size: 词汇表大小
    :param num_hiddens: 隐藏单元个数
    :param device:
    :return: 模型中需要使用的各种参数 W_xh, W_hh, b_h, W_hq, b_q
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # torch.randn(*size, device)
        # 根据 正态分布 生成随机数，填充形状为 size 的张量
        #
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # print(b_h.shape) # torch.Size([512]) num_hiddens

    # 输出层
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # print(b_q.shape) # torch.Size([28]) len(vocab)

    params = [W_xh, W_hh, b_h, W_hq, b_q]

    # 附加梯度
    # 将每一个参数添加到计算图中
    for param in params:
        param.requires_grad_(True)
    return params
```

#### 2 RNN模型初始化

```python
def init_rnn_state(batch_size, num_hidden, device):
    """
    设置 rnn 的
    :param batch_size: 批量大小
    :param num_hidden: 隐藏单元数
    :param device:
    :return: 一个用 0 填充的张量 H，shape (batch_size, num_hidden)，为了防止后面隐状态包含多个变量的情况，这里使用元组将这个张量包起来
    """
    return (torch.zeros((batch_size, num_hidden), device=device),)
```

#### 3 设置RNN模型

```python
def rnn(input, state, params):
    """
    在一个时间步内，计算隐藏状态和输出，也就是神经网络
    :param input: 输入 shape (num_steps, batch_size, vocab_size)
    :param state: 隐藏状态
    :param params: rnn 计算的各种参数 W_xh, W_hh, b_h, W_hq, b_q
    :return: 输出 和 当前隐藏状态
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X shape (batch_size, vocab_size)
    for X in input:
        # H 是隐藏状态， 这里使用 tanh 函数作为激活函数
        # torch.mm(X, W_xh) 也就是矩阵 X 和 矩阵 W_xh 相乘
        # 也就是下一行代码等于 X * W_xh + H * W_hh + b_h
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # print(H.shape) # torch.Size([2, 512]) batch_size, num_hiddens
        # Y 是输出
        Y = torch.mm(H, W_hq) + b_q
        # print(b_q.shape) # torch.Size([28])
        # print(Y.shape) # torch.Size([2, 28]) batch_size, vocab_size
        outputs.append(Y)

    # A = torch.cat(Bs, dim=i)
    # Bs 被拼接的序列
    # dim 扩维序号，就是选择扩张的维度，在 [0, len(Bs)] 之间
    # 当 dim = 0 时，也就是拼接是按照扩张第 0 维度来进行的，也就是拼接的结果会扩张行维度
    # 针对于例子，也就是会将 Bs 中的元素按照列堆起来生成 A
    # print(torch.cat(outputs, dim=0).shape) # torch.Size([10, 28])
    return torch.cat(outputs, dim=0), (H,)
```

#### 4 对上述步骤的包装

```python
class RNNModelScratch:  # @save
    """
    对上述操作的包装
    """

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """

        :param vocab_size: 词汇表大小
        :param num_hiddens: 隐藏单元大小
        :param device:
        :param get_params: 获取参数的函数
        :param init_state: 初始神经网络状态的函数
        :param forward_fn: 神经网络函数
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        对 RNNModelScratch 类对象调用时，会自动调用该 __call__ 函数
        
        将 X 转化为其对应的 one_hot 编码, 类型为 torch.float32"""""
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """获取初始状态"""
        return self.init_state(batch_size, self.num_hiddens, device)
```

5 设置模型

```python
num_hiddens = 512
# 生成 net 对象，调用 RNNModelScratch 中的 __init__ 函数
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
```



```python

# 生成初始隐藏状态 ,调用 begin_state 函数，也就是 init_rnn_state() 函数
state = net.begin_state(X.shape[0], d2l.try_gpu())
#  调用 RNNModelScratch 中的 __call__ 函数，也就是 rnn() 函数
# Y 是当前时间步的输出，new_state 是当前时间步的隐藏状态
Y, new_state = net(X.to(d2l.try_gpu()), state)
# print(Y.shape, len(new_state), new_state[0].shape)
# torch.Size([10, 28]) 1 torch.Size([2, 512])

```



#### 完整代码

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from matplotlib import pyplot as plt

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

"""
one-hot 编码
"""
one_hot_0_2 = F.one_hot(torch.tensor([0, 2]), len(vocab))
# print(one_hot_0_2)
# len(vocab) == 28
X = torch.arange(10).reshape((2, 5))


# print(F.one_hot(X))
# print(F.one_hot(X.T, len(vocab)).shape)

# for x in X.T: x shape (2, 28) (batch_size, len(X.T))

"""
绘图函数和类
"""
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.legend = legend
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xscale = xscale
    plt.yscale = yscale
    if legend:
        axes.legend(legend)
    axes.grid()

class MyPlotMethod:
    def __init__(self,  xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        plt.rcParams['figure.figsize'] = figsize
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

def get_params(vocab_size, num_hiddens, device):
    """
    生成模型计算需要的各种参数

    模型流程
    1. 计算当前层的初始隐藏状态：H_o = X * W_xh + H * W_hh + b_h
    2. 将初始隐藏状态经过激活函数转化成为真正的当前隐藏状态：H = 激活函数(H_o)
    3. 计算输出：Y = H * W_hq + b_q

    最后计算出的 H shape (batch_size, num_hiddens)
    Y shape (batch_size, vocab_size)

    X: inputs shape (batch_size, vocab_size)
    W_xh: inputs 的权重 shape (vocab_size, num_hiddens)
    H: hidden units shape (batch_size, num_hiddens)
    W_hh: 隐藏状态的权重 shape (num_hiddens, num_hiddens)
    b_h: 计算当前隐藏状态需要的偏差 shape num_hiddens
    new_H:  shape (batch_size, num_hiddens)
    W_hq: 输出层权重 shape(num_hiddens, vocab_size)
    b_q: 输出层偏差 shape vocab_size
    Y: outputs (batch_size, vocab_size) 形状又变成和 inputs 相同的形状
    :param vocab_size: 词汇表大小
    :param num_hiddens: 隐藏单元个数
    :param device:
    :return: 模型中需要使用的各种参数 W_xh, W_hh, b_h, W_hq, b_q
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # torch.randn(*size, device)
        # 根据 正态分布 生成随机数，填充形状为 size 的张量
        #
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # print(b_h.shape) # torch.Size([512]) num_hiddens

    # 输出层
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # print(b_q.shape) # torch.Size([28]) len(vocab)

    params = [W_xh, W_hh, b_h, W_hq, b_q]

    # 附加梯度
    # 将每一个参数添加到计算图中
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hidden, device):
    """
    设置 rnn 的
    :param batch_size: 批量大小
    :param num_hidden: 隐藏单元数
    :param device:
    :return: 一个用 0 填充的张量 H，shape (batch_size, num_hidden)，为了防止后面隐状态包含多个变量的情况，这里使用元组将这个张量包起来
    """
    return (torch.zeros((batch_size, num_hidden), device=device),)


def rnn(input, state, params):
    """
    在一个时间步内，计算隐藏状态和输出，也就是神经网络
    :param input: 输入 shape (num_steps, batch_size, vocab_size)
    :param state: 隐藏状态
    :param params: rnn 计算的各种参数 W_xh, W_hh, b_h, W_hq, b_q
    :return: 输出 和 当前隐藏状态
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X shape (batch_size, vocab_size)
    for X in input:
        # H 是隐藏状态， 这里使用 tanh 函数作为激活函数
        # torch.mm(X, W_xh) 也就是矩阵 X 和 矩阵 W_xh 相乘
        # 也就是下一行代码等于 X * W_xh + H * W_hh + b_h
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # print(H.shape) # torch.Size([2, 512]) batch_size, num_hiddens
        # Y 是输出
        Y = torch.mm(H, W_hq) + b_q
        # print(b_q.shape) # torch.Size([28])
        # print(Y.shape) # torch.Size([2, 28]) batch_size, vocab_size
        outputs.append(Y)

    # A = torch.cat(Bs, dim=i)
    # Bs 被拼接的序列
    # dim 扩维序号，就是选择扩张的维度，在 [0, len(Bs)] 之间
    # 当 dim = 0 时，也就是拼接是按照扩张第 0 维度来进行的，也就是拼接的结果会扩张行维度
    # 针对于例子，也就是会将 Bs 中的元素按照列堆起来生成 A
    # print(torch.cat(outputs, dim=0).shape) # torch.Size([10, 28])
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:  # @save
    """
    对上述操作的包装
    """

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """

        :param vocab_size: 词汇表大小
        :param num_hiddens: 隐藏单元大小
        :param device:
        :param get_params: 获取参数的函数
        :param init_state: 初始神经网络状态的函数
        :param forward_fn: 神经网络函数
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        对 RNNModelScratch 类对象调用时，会自动调用该 __call__ 函数
        
        将 X 转化为其对应的 one_hot 编码, 类型为 torch.float32"""""
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """获取初始状态"""
        return self.init_state(batch_size, self.num_hiddens, device)


num_hiddens = 512
# 生成 net 对象，调用 RNNModelScratch 中的 __init__ 函数
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
# 生成初始隐藏状态 ,调用 begin_state 函数，也就是 init_rnn_state() 函数
state = net.begin_state(X.shape[0], d2l.try_gpu())
#  调用 RNNModelScratch 中的 __call__ 函数，也就是 rnn() 函数
# Y 是当前时间步的输出，new_state 是当前时间步的隐藏状态
Y, new_state = net(X.to(d2l.try_gpu()), state)
# print(Y.shape, len(new_state), new_state[0].shape)
# torch.Size([10, 28]) 1 torch.Size([2, 512])


"""预测"""


def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    进行预测的函数
    :param prefix: 需要预测的字符串
    :param num_preds: 要预测步长，要预测多少步
    :param net: 神经网络
    :param vocab: 词汇表
    :param device:
    :return: 预测到字符串
    """
    # 初始状态 H_o shape (batch_size, num_hiddens)
    state = net.begin_state(batch_size=1, device=device)
    # print(state[0].shape) torch.Size([1, 512])
    # vocab[prefix[0]]，也就是 vocab[c] 调用 vocab 类中的 __getitem__() 方法，将字符转换成其在 vocab 中对应的索引
    outputs = [vocab[prefix[0]]]  # 将 prefix 第 0 位加入 outputs
    # print(''.join([vocab.idx_to_token[i] for i in outputs])) # t
    # 匿名函数 获取 outputs 列表中的最后一位
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热部分，将 prefix 挨个投入到 net 中做计算，并将每一位加入到 outputs 中
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # print(''.join([vocab.idx_to_token[i] for i in outputs])) # time traveller
    # 预测部分，进行 num_preds 步预测
    # 每次获得输出张量中最大值的索引，然后将该索引加入到 outputs 中作为预测结果
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # y.argmax(dim=1) 获得张量中最大值的索引
        # dim=1 就是按第 1 维度进行，就是取列中最大值
        # print(y.shape) # torch.Size([1, 28])
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


a = predict_ch8('time traveller', 10, net, vocab, d2l.try_gpu())


# print(a) # time travellerrrrrrrrrrr || time travellermentqjsvme 因为没有训练网络，所以会生成很荒谬的结果


def grad_clipping(net, theta):
    """
    梯度裁剪

    问题：当时间步长 T 较大时，导致数值不稳定，例如导致梯度爆炸或梯度消失。
    目标：保证模型的稳定性
    基本思路：保证梯度每次下降的量在一个安全范围内，比如确定一个范围，当梯度的值超出这个梯度时，则通过某些方式，将其折叠到这个安全范围内
    具体方法：
    1. 直接裁剪：当梯度大于阈值时，将梯度裁剪到最大阈值，当梯度小于阈值时，将梯度裁剪到最小阈值
    2. 比例减少：通过将梯度单位向量乘以阈值来裁剪梯度
    3. 结合上述两种方式，g <- min(1, theta / ||g||) * g，当梯度在阈值范围内时，就直接使用阈值，如果梯度大于阈值，则将梯度裁剪为阈值乘以梯度方向上的单位向量
    :param net: 网络
    :param theta: 阈值
    :return: void
    """
    # isinstance 判断是一个对象是否是一个已知对象
    # 下面的选择结构是得到参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 求 |p|，是将所有的梯度元素拼接成一个向量
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # theta 就是阈值，如果梯度大于阈值，就对梯度进行裁剪
    if norm > theta:
        for param in params:
            # p = p * theta / |p| 梯度裁剪
            param.grad[:] *= theta / norm


"""
训练部分

定义函数在一个迭代周期内训练模型

针对于不同的采样方式，采取不同的隐藏状态初始化方法
1. 针对于顺序抽样：
    1. 当前小批量数据最后一个样本的隐藏状态，将用于初始化下一个小批量数据第一个样本的隐藏状态。
    2. 将每一个隐藏状态的计算都限制在每一个小批量样本中——为了降低计算量
2. 针对于随机抽样：
    对每一个周期重新初始化隐藏状态。
"""


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """
    训练网络一个迭代周期

    过程：
    1. 对隐状态的处理
        1. 对于顺序采样初始隐状态和随机抽样每个 epoch 的初始隐状态的初始化
        2. 对于其他隐状态，进行 detach_() 操作，就是将这个隐状态从原本其所在的计算图中剥离出来
    2. 获取训练数据 X 和标签 Y
    3. 对 X 计算预测值 y_hat 和隐状态 state
    4. 使用 y 和 y_hat 计算损失函数 loss
    5. 更新参数，优化模型：
        1. 优化器初始化
        2. 对损失函数计算梯度
        3. 梯度裁剪
        4. 根据梯度更新参数

    :param net: 神经网络
    :param train_iter: 训练迭代器，训练数据也在 train_iter
    :param loss: 损失函数
    :param updater: 优化函数
    :param device: 设备
    :param use_random_iter: 是否使用随机采样的标志
    :return:
    """
    state, timer = None, d2l.Timer()

    # class Accumulator:
    #     """For accumulating sums over `n` variables."""
    #     def __init__(self, n):
    #         """Defined in :numref:`sec_softmax_scratch`"""
    #         self.data = [0.0] * n
    #
    #     def add(self, *args):
    #         self.data = [a + float(b) for a, b in zip(self.data, args)]
    #
    #     def reset(self):
    #         self.data = [0.0] * len(self.data)
    #
    #     def __getitem__(self, idx):
    #         return self.data[idx]
    metric = d2l.Accumulator(2)  # 用来保存训练损失之和，词元数量的数据结构
    for X, Y in train_iter:
        # 对于相邻采样的第一个样本，或者是随机抽样的每次分区，都需要初始化隐藏状态
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 将 state 从计算图中剥离出，设置成叶子tensor
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        # 将 X, y 投指定计算资源
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state) # 神经网络计算
        l = loss(y_hat, y.long()).mean() # 计算损失
        # 求导 参数更新
        # 判断 updater 使用的是模块内置的更新器还是自己写的
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward() 
            grad_clipping(net, 1)
            updater(batch_size=1)
        # 因为前面的 loss，取了均值，也就是 l = loss(y_hat, y.long()).mean()，所以这里再将 l * y.numel()
        # list.numel() 返回数组中元素的个数
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """
    训练模型
    :param net: 神经网络
    :param train_iter: 训练数据
    :param vocab: 词汇表
    :param lr: 学习率
    :param num_epochs: 周期
    :param device: 计算资源
    :param use_random_iter: 采样方式标志
    :return:
    """
    loss = nn.CrossEntropyLoss()
    # 绘制图形
    animator = MyPlotMethod(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    plt.show()



num_epochs, lr = 500, 1

"""相邻采样"""
# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


"""随机抽样"""
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)
```

## 2 简洁实现

代码

```python
```

