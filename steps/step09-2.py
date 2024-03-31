'''
9.2 backward method 간소화
역전파 시 번거로움을 줄이기 위한 것. y.grad = np.array(1.0)을 생략
'''

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [ self.creator ]
        while funcs:
            f = funcs.pop()     # 함수를 가져온다.
            x, y = f.input, f.output    # 함수의 입출력을 가져온다.
            x.grad = f.backward(y.grad) # backward 함수를 호출한다.
            
            if x.creator is not None:
                funcs.append(x.creator)     # 하나 앞의 함수를 추가한다.


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)
    
if __name__ == "__main__":

    x = Variable(np.array(0.5))
    # a = square(x)
    # b = exp(a)
    # y = square(b)
    y = square(exp(square(x)))  # 연속하여 적용

    # backward
    # y.grad = np.array(1.0)
    y.backward()
    print(x.grad)