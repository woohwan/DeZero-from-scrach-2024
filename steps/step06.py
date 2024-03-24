import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data    # 통상 값
        self.grad = None    # 미분 값: multivariate differential --> gradient

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input      # 입력 변수를 기억(보관)한다
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x**2
    
    # gy: ndarray instance, 출력 쪽에서 전해지는 미분 값을 전달하는 역할
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx

class Exp(Function):
    # local gradient * upstream
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    print(x.grad)