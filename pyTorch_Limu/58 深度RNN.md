# 深度循环神经网络

目的：为了获得更多非线性性

操作：增加更多层隐藏层

## 0 导入相关依赖库和数据


```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## 1 获取模型参数方法


```python
vocab_size = len(vocab)
num_inputs, num_hiddens, num_layers, device = vocab_size, 256, 2, d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, vocab_size)
model = model.to(device)

num_epochs, lr = 500, 2

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

```

    time travelleryou can show black is white by argument said filby
    


    
![svg](./58%20深度RNN/main_4_1.svg)