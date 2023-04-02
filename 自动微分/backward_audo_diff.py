from typing import List, NamedTuple, Callable, Dict, Optional

import numpy as np

_name = 1


# fresh_name 用于打印跟 tape 相关的变量，并用 _name 来记录是第几个变量
def fresh_name():
    global _name
    name = f"v{_name}"
    _name += 1
    return name


# 代码中添加了变量类 Variable 来跟踪计算梯度，并添加了梯度函数 grad() 来计算梯度。
class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_name()

    # repr=representation, 打印对象出来的值
    def __repr__(self):
        return repr(self.value)

    @staticmethod
    def constant(value, name=None):
        var = Variable(value, name)
        print(f"{var.name} = {var.value}")
        return var

    # 这里面只提供了 乘、加、减、sin、log 五种计算方式
    def __mul__(self, other):
        return ops_mul(self, other)

    def __add__(self, other):
        return ops_add(self, other)

    def __sub__(self, other):
        return ops_sub(self, other)

    def sin(self):
        return ops_sin(self)

    def log(self):
        return ops_log(self)


# NamedTuple 是一个 Python 内置类型，它可以用来创建具有特定属性和类型注释的不可变数据类。
# Callable: 用于声明可调用对象的类型
class Tape(NamedTuple):
    inputs: List[str]
    outputs: List[str]
    propagate: Callable[[List[Variable]], List[Variable]]


gradient_tape = []


# reset_tape
def reset_tape():
    global _name
    _name = 1
    gradient_tape.clear()


def ops_mul(self, other):
    # forward
    x = Variable(self.value * other.value)
    print(f"{x.name} = {self.name} * {other.name}")

    # backward
    # x = self * other
    # 在执行propagate函数时，里面也涉及variable基本运算，仍然会产生tape记录, 需要reset_tape
    def propagate(dl_outputs):
        # dl_dx,为将dl_outputs列表第一个元素赋值给dl_dx
        dl_dx, = dl_outputs
        dx_dself = other
        dx_dother = self
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs

    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_add(self, other):
    # forward
    x = Variable(self.value + other.value)
    print(f"{x.name} = {self.name} + {other.name}")

    # backward
    # x = self * other
    def propagate(dl_outputs):
        dl_dx, = dl_outputs
        dx_dself = Variable(1.)
        dx_dother = Variable(1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs

    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_sub(self, other):
    # forward
    x = Variable(self.value - other.value)
    print(f"{x.name} = {self.name} - {other.name}")

    # backward
    # x = self * other
    def propagate(dl_outputs):
        dl_dx, = dl_outputs
        dx_dself = Variable(1.)
        dx_dother = Variable(-1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs

    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_sin(self):
    # forward
    x = Variable(np.sin(self.value))
    print(f"{x.name} = sin({self.name})")

    # backward
    # x = self * other
    def propagate(dl_outputs):
        dl_dx, = dl_outputs
        dx_dself = Variable(np.cos(self.value))
        dl_dself = dl_dx * dx_dself
        dl_dinputs = [dl_dself]
        return dl_dinputs

    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_log(self):
    # forward
    x = Variable(np.log(self.value))
    print(f"{x.name} = log({self.name})")

    # backward
    # x = self * other
    def propagate(dl_outputs):
        dl_dx, = dl_outputs
        dx_dself = Variable(1 / self.value)
        dl_dself = dl_dx * dx_dself
        dl_dinputs = [dl_dself]
        return dl_dinputs

    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


# grad 呢是将变量运算放在一起的梯度函数，函数的输入是 l 和对应的梯度结果 results。
def grad(l, results):
    # l对所有结点的偏导数
    dl_d = {}
    # 输出对自身的梯度
    dl_d[l.name] = Variable(1.)
    print("dl_d", dl_d)

    # 获取输出节点梯度作为propagate输入
    def gather_grad(entries):
        return [dl_d[entry] if entry in dl_d else None for entry in entries]

    for entry in reversed(gradient_tape):
        print(entry)
        # 对于l只有一个输出节点
        dl_doutputs = gather_grad(entry.outputs)
        dl_dinputs = entry.propagate(dl_doutputs)

        for input, dl_input in zip(entry.inputs, dl_dinputs):
            if input not in dl_d:
                dl_d[input] = dl_input
            # 计算某节点的累积梯度
            else:
                dl_d[input] += dl_input

    for name, value in dl_d.items():
        print(f"d{l.name}_d{name} = {value.name}")

    return gather_grad([result.name for result in results])
    # return [dl_d[result.name] for result in results]


if __name__ == '__main__':
    # 所有变量类型都为variable，类似pytorch的tensor
    # 重置tape记录
    reset_tape()
    # forward
    x1 = Variable.constant(value=2.0, name="v_-1")
    x2 = Variable.constant(value=5.0, name="v0")
    # f = Variable.log(x1) + x1 * x2 - Variable.sin(x2)
    f = x1.log() + x1 * x2 - x2.sin()
    print(f)
    # tape 记录的计算过程和反向传播函数
    print(gradient_tape)
    dx1, dx2 = grad(f, [x1, x2])
    print(f"dx1 = {dx1}")
    print(f"dx2 = {dx2}")
