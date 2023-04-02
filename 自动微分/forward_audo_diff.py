import numpy as np
import torch


class ADTangent:
    """前向自动微分"""
    def __init__(self, x, dx):
        self.x = x
        self.dx = dx

    # print ADTangent对象时会输出str
    def __str__(self):
        content = f"value:{self.x:.4f}, grad:{self.dx}"
        return content

    def __add__(self, other):
        if isinstance(other, ADTangent):
            x = self.x + other.x
            dx = self.dx + other.dx
        elif isinstance(other, float):
            x = self.x + other
            dx = self.dx
        else:
            NotImplementedError
        return ADTangent(x, dx)

    def __sub__(self, other):
        if isinstance(other, ADTangent):
            x = self.x - other.x
            dx = self.dx - other.dx
        elif isinstance(other, float):
            x = self.x - other
            dx = self.dx
        else:
            NotImplementedError
        return ADTangent(x, dx)

    def __mul__(self, other):
        if isinstance(other, ADTangent):
            x = self.x * other.x
            dx = self.dx * other.x + self.x * other.dx
        elif isinstance(other, float):
            x = self.x * other
            dx = self.dx * other
        else:
            NotImplementedError
        return ADTangent(x, dx)

    # 对自身的操作，涉及一个对象
    def log(self):
        x = np.log(self.x)
        dx = 1 / self.x * self.dx
        return ADTangent(x, dx)

    def sin(self):
        x = np.sin(self.x)
        dx = np.cos(self.x) * self.dx
        return ADTangent(x, dx)


if __name__ == '__main__':
    # 使用ADTangent重载操作符
    # sin和log是静态方法可以这样写
    x1 = ADTangent(2, 1)
    x2 = ADTangent(5, 0)
    f = ADTangent.log(x1) + x1 * x2 - ADTangent.sin(x2)
    print(f)

    # 正解，sin和log不是静态方法
    f = x1.log() + x1 * x2 - x2.sin()
    print(f)

    # 使用pytorch
    x1 = torch.tensor(2.0, requires_grad=True)
    x2 = torch.tensor(5.0, requires_grad=True)
    f = torch.log(x1) + x1 * x2 - torch.sin(x2)
    f.backward()
    print(f, x1.grad, x2.grad)
