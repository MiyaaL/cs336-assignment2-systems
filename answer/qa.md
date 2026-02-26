# Problem And Answer

## 1.1.3 benchmarking_script

### a

见benchmark.py

### b

测试详细数据见表，这里不详述。backward的耗时大约是forward的两倍，因为做了warm-up，测试时间波动不大

### c

去掉warm-up结果会有波动，耗时会变长

## 1.1.4 nsys_profile

### a

with nvtx.range 会更准一点，计时会更短一点，并且带来的 overhead 几乎不会影响到外层的 time 计时（因为循环的次数并不是很多）。

### b

大部分都是 ampere_sgemm 算子主导，之后会有一个 elementwise 的小热点算子。大部分的调用次数很难和迭代次数直接对上。

### c

elementwise reduce_kernel 等等

### d

forward only里面 gemm 占比会明显更高，其他 kernel 占比更低。

### e

softmax 耗时占比并不像 FLOPS 那样占比那么小，在某些规模下两者耗时处于一个量级

## 1.1.5 mixed_precision_accumulation

```python
import torch

s=torch.tensor(0,dtype=torch.float32)
for _ in range(1000):
    s+= torch.tensor(0.01,dtype=torch.float32)
print(s) # tensor(10.0001)
s=torch.tensor(0,dtype=torch.float16)
for _ in range(1000):
    s+= torch.tensor(0.01,dtype=torch.float16)
print(s) # tensor(9.9531, dtype=torch.float16)
s=torch.tensor(0,dtype=torch.float32)
for _ in range(1000):
    s+= torch.tensor(0.01,dtype=torch.float16)
print(s) # tensor(10.0021)
s=torch.tensor(0,dtype=torch.float32)
for _ in range(1000):
    x=torch.tensor(0.01,dtype=torch.float16)
    s+= x.type(torch.float32)
print(s) # tensor(10.0021)
```

- 高精度 + 低精度，低精度会做隐式类型转换，转换过程可能会引入误差

### a

```python
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("after fc1:", x.dtype)
        x = self.ln(x)
        print("after ln:", x.dtype)
        x = self.fc2(x)
        print("after fc2:", x.dtype)
        return x

def main():
    device = torch.device("cuda")             # 用 GPU
    model = ToyModel(10, 10).to(device)       # 模型放到 GPU
    x = torch.randn(10, 10, device=device)    # 输入也在 GPU

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        print("y:", y.dtype)

    print("fc1 weight:", model.fc1.weight.dtype)
    print("fc2 weight:", model.fc2.weight.dtype)
    print("ln weight:", model.ln.weight.dtype)
    print("ln bias:", model.ln.bias.dtype)

if __name__ == "__main__":
    main()
```

输出

```txt
after fc1: torch.float16
after ln: torch.float32
after fc2: torch.float16
y: torch.float16
fc1 weight: torch.float32
fc2 weight: torch.float32
ln weight: torch.float32
ln bias: torch.float32
```

可以发现，autocast 只在计算 Linear 时会自动进行低精度计算，而 layernorm 则不会。

### b

回忆 RMSNorm 的计算公式（这里虽然是layernorm但是也同理）：

$$
    RMSNorm(x) = \frac {x}{\sqrt{\frac{1}{n} \sum_{i=1}^n a_i^2 + \epsilon}}
$$

分母部分的累加决定了必须使用更高的精度来保证结果准确。

如果使用 bfloat16(E8M7)，对比 float16(E5M10)，尾数部分还缩短了，也会有误差。因此 layernorm 部分也不能用 bfloat16。

### c

换成混合精度后，耗时可能会减少一倍以上。具体可以benchmark测试得到数据。

## 1.1.6 memory_profiling

### a

这里 A100 跑不了 2.7B 的模型，以 xl 代替。峰值部分在 optimizer.step() 部分

### b

略

### c

xl 模型的内存使用峰值在 29.8GiB (full training)，直接使用混合精度甚至可能会增加内存占用，主要是会让现存碎片化，增加类型转化的额外开销。

### d

xl model: 4 * 128 * 1600 / 1024 / 1024 = 0.78125 MiB

### e

主要是优化器的内存占用