from activations import ReLU, Tanh, Sigmoid
from layers import Dropout, BatchNorm, DenseLayer


def activations():
    x = [[[-1.0, 1.0, 2.0, 3.0],
         [1.0, 4.0, -2.5, 3.0]],

         [[-1.0, 1.0, 2.0, 3.0],
          [1.0, 4.0, -2.5, 3.0]]]

    print('Original X =', x)

    relu = ReLU()

    print('ReLU(x) =', relu.forward(x))
    print('ReLU`(x) =', relu.backward(x))

    tanh = Tanh()

    print('Tanh(x) =', tanh.forward(x))
    print('Tanh`(x) =', tanh.backward(x))

    sigmoid = Sigmoid()

    print('Sigmoid(x) =', sigmoid.forward(x))
    print('Sigmoid`(x) =', sigmoid.backward(x))


def dropout():
    x = [[[-1.0, 1.0, 2.0, 3.0],
          [1.0, 4.0, -2.5, 3.0]],

         [[-1.0, 1.0, 2.0, 3.0],
          [1.0, 4.0, -2.5, 3.0]],
         ]

    dropout = Dropout(input_size=2, output_size=3)

    print('Dropout forward: ', dropout.forward(x))

    dropout.eval()

    print('Dropout backward: ', dropout.backward(x))


def batchnorm():
    x = [[1.0, 1.0, 1.0],

         [2.0, 2.0, 2.0],

         [3.0, 3.0, 3.0]]

    bn = BatchNorm()

    print('BatchNorm forward: ', bn.forward(x))
    bn.eval(True)
    print('BatchNorm forward eval: ', bn.forward(x))
    print('BatchNorm running_covar', bn.running_covariance)
    print('BatchNorm running_mean', bn.running_mean)

    print('BatchNorm backward: ', bn.backward(x))


def denselayer():
    pass


# activations()
# dropout()
# batchnorm()
# denselayer()
