import torch
import torch.nn as nn

# 初始化一个ReLU激活函数
relu = nn.ReLU()

# 创建一个张量
x = torch.randn(4, 4)

# 应用ReLU激活函数
output = relu(x)


# 初始化一个Sigmoid激活函数
sigmoid = nn.Sigmoid()

# 应用Sigmoid激活函数
output = sigmoid(x)


# 初始化一个Tanh激活函数
tanh = nn.Tanh()

# 应用Tanh激活函数
output = tanh(x)

# 初始化一个Leaky ReLU激活函数，设置negative_slope参数
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# 应用Leaky ReLU激活函数
output = leaky_relu(x)


# 初始化一个ELU激活函数，可以设置alpha参数
elu = nn.ELU(alpha=1.0)

# 应用ELU激活函数
output = elu(x)

# 初始化一个SELU激活函数
selu = nn.SELU()

# 应用SELU激活函数
output = selu(x)

# 初始化一个Softmax激活函数，通常用于多分类问题
softmax = nn.Softmax(dim=1)  # dim参数指定对哪个维度进行Softmax操作

# 应用Softmax激活函数
output = softmax(x)

# 初始化一个LogSoftmax激活函数，通常用于多分类问题
log_softmax = nn.LogSoftmax(dim=1)

# 应用LogSoftmax激活函数
output = log_softmax(x)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
