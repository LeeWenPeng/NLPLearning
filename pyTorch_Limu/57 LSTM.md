# LSTM 的实现

## 0 导入环境和数据


```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## 1 初始化模型参数
根据 LSTM 公式
1. 遗忘门 Forout gate

$$
        F_t = \sigma(X_tW_{xf} + H_{t-1}W_{hf} + b_f)
$$

2. 输入门 Input gate
   
   $$
    I_t = \sigma(X_tW_{xi} + H_{t-1}W_{hi} + b_i)
   $$

3. 输出门 Output gate
   
   $$
    O_t = \sigma(X_tW_{xO} + H_{t-1}W_{hO} + b_o)
   $$

4. 候选记忆单元 `C^{hat}_t`

$$
        C^{hat}_t = \tanh(X_tW_{xc} + H_{t-1}W_{hc} + b_c)
$$        

每个公式都需要三个参数，共需要十二个参数。
由于三个门和候选记忆单元的形状都和隐藏状态形状一样，为[num_inputs, num_hiddens]，所以三个参数的形状也一样：
1. 第一个参数形状 [inputs_size, num_hiddens]
2. 第二个参数形状 [num_hiddens, num_hiddens]
3. 第三个参数形状 [num_hiddens]

输出层

$$
output_t = \sigma (Y_t W_hq + b_q)
$$

输出层需要两个参数:
+ $W_q$: 形状为 (num_hiddens, output_size)
+ $b_q$: 形状为 (output_size)


```python
def get_lstm_params(vocab_size, num_hiddens, device):

    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device)
        )
    
    W_xf, W_hf, b_f = three()
    W_xi, W_hi, b_i = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs,device=device)

    params = [W_xf, W_hf, b_f,
              W_xi, W_hi, b_i,
              W_xo, W_ho, b_o,
              W_xc, W_hc, b_c,
              W_hq, b_q]

    # 附加梯度
    for param in params:
        param.requires_grad_(True)

    return params
```

## 2 定义模型

### 2.1 定义初始隐藏状态

根据 LSTM 在时间步骤 t 计算隐藏状态 $H_t$ 的过程：

1. 计算候选记忆单元

    $$
    C^{hat}_t = \tanh(X_tW_{xc} + H_{t-1}W_{hc} + b_c)
    $$

2. 计算记忆单元

    $$
    C_t = F_t \odot C_{t-1} + I_t \odot C^{hat}_t
    $$

3. 计算隐藏状态

    $$
    H_t = O_t \odot \tanh(C_t)
    $$

所以当计算 $H_1$ 时，需要初始的记忆单元 $C_0$ 和初始的隐藏状态 $H_0$

所以设置隐藏状态的初始函数时，需要返回的量有**初始记忆单元**和**初始隐藏状态**


```python
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

### 2.2 定义实际的模型

包括隐藏层和输出层。

隐藏层就是上述计算隐藏状态 $H_t$ 的过程

输出层：

$$
Y = H_tW_{hq} + b_q
$$

```python
def lstm(inputs, state, params):

    # 获取参数
    [W_xf, W_hf, b_f,
     W_xi, W_hi, b_i,
     W_xo, W_ho, b_o,
     W_xc, W_hc, b_c,
     W_hq, b_q] = params
    # 获取最初的隐藏状态和记忆单元
    (H, C )= state
    # 定义保存 output 的数据结构
    outputs = []

    for X in inputs:

        # 隐藏层
        # 生成几个门
        I_t = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F_t = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O_t = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        # 候选记忆单元
        C_hat = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        # 记忆单元
        C = F_t * C + I_t * C_hat
        # 隐藏状态
        H = O_t * torch.tanh(C)
        
        # 输出层
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

## 3 训练与预测


```python
vocab_size, num_hiddens, device=len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(vocab_size, num_hiddens, device=device, get_params=get_lstm_params, init_state=init_lstm_state, forward_fn=lstm)

d2l.train_ch8(model, train_iter, vocab, lr=lr, num_epochs=num_epochs, device=device, use_random_iter=False)
```

    perplexity 1.1, 30445.6 tokens/sec on cuda:0
    time traveller but now you begin to seethe obomet of all that ul       
    travelleryou can show black is white by argument said filby
    


    
![svg](./57%20LSTM/main_files/main_10_1.svg)
    


## 4 完整代码


```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 定义模型参数设置函数
def get_lstm_params(vocab_size, num_hiddens, device):

    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device)
        )
    
    W_xf, W_hf, b_f = three()
    W_xi, W_hi, b_i = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs,device=device)

    params = [W_xf, W_hf, b_f,
              W_xi, W_hi, b_i,
              W_xo, W_ho, b_o,
              W_xc, W_hc, b_c,
              W_hq, b_q]

    # 附加梯度
    for param in params:
        param.requires_grad_(True)

    return params

# 定义初始化函数
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

# 定义 lstm 模型
def lstm(inputs, state, params):

    # 获取参数
    [W_xf, W_hf, b_f,
     W_xi, W_hi, b_i,
     W_xo, W_ho, b_o,
     W_xc, W_hc, b_c,
     W_hq, b_q] = params
    # 获取最初的隐藏状态和记忆单元
    (H, C )= state
    # 定义保存 output 的数据结构
    outputs = []

    for X in inputs:

        # 隐藏层
        # 生成几个门
        I_t = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F_t = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O_t = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        # 候选记忆单元
        C_hat = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        # 记忆单元
        C = F_t * C + I_t * C_hat
        # 隐藏状态
        H = O_t * torch.tanh(C)
        
        # 输出层
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

# 训练和预测
vocab_size, num_hiddens, device=len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(vocab_size, num_hiddens, device=device, get_params=get_lstm_params, init_state=init_lstm_state, forward_fn=lstm)

d2l.train_ch8(model, train_iter, vocab, lr=lr, num_epochs=num_epochs, device=device, use_random_iter=False)
```

## 4 LSTM 的简单实现


```python
from d2l import torch as d2l
import torch
from torch import nn

# 参数设置
train_iter, vocab = d2l.load_data_time_machine(batch_size=32, num_steps=35)
vocab_size = len(vocab)
num_inputs, num_hiddens, device = vocab_size, 256, d2l.try_gpu()
num_epochs, lr = 500, 1

lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

    perplexity 1.0, 313135.6 tokens/sec on cuda:0
    time travelleryou can show black is white by argument said filby
    travelleryou can show black is white by argument said filby
    


    
![svg](./57%20LSTM/main_files/main_14_1.svg)
    

