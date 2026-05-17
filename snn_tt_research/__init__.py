"""Research package: Tensor-Train acceleration of Spiking Neural Networks.

Three interconnected research blocks:
    block1_iris            small-scale Iris demo with Gaussian receptive fields
    block2_mnist_stdp      MNIST STDP encoder with post-hoc TT compression
    block3_fashion_surgrad Fashion-MNIST surrogate-gradient MLP with TT and LowRank
"""

__all__ = ["common", "block1_iris", "block2_mnist_stdp", "block3_fashion_surgrad"]
