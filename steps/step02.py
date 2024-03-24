import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data  # 데이터를 꺼낸다.
        y = self.forward(x)     # 구체적인 계산은 forward에서 한다.
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
if __name__ == "__main__":
    x = Variable(10)
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)