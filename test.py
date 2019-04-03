from jax import jit
import jax.numpy as np

def f(x, y):
    return x + y

def f1(x, y):
    return x * x + y * y

def f2(x, y, z):
    return x + y * z

def f3(params, inputs):  # The predict() fn from "How it Works"
    print(params)
    print(inputs)
    for W, b in params:
        z = np.dot(inputs, W) + b
        inputs = np.tanh(z)
    return z

def f4(xs):
    res = 0
    for x in xs:
        res += x
    return res

def f5(n):
    res = 0
    for i in range(n[0]):
        res += i
    return res

def f6(n):
    res = 0
    for i in range(n):
        res += i
    return res

if __name__ == '__main__':
    f3_args1 = [[(100, 1)], 42]
    f3_args2 = [[(100, 1), (1000, 2)], 42]
    f3_args3 = [[(100, 1), (1000, 2), (10000, 3)], 42]

    fns = {
        f: (2, 3),
        f1: (2, 3),
        f2: (2, 3, 4),
        f3: f3_args1,
        f3: f3_args2,
        f3: f3_args3,
        f4: (list(range(5)),),
        # f5: ([5],),
        # f6: (5,),
    }

    print("starting--------------")
    for fn, arg in fns.items():
        print(str(fn.__name__) + "------------------")
        jitted_f = jit(fn)
        print("jitted--------------------")
        print("result: " + repr(jitted_f(*arg)))
