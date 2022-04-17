import os
import pickle
import random


class DataLoader:
    def __init__(self, seed: int = 42):
        pass

    @staticmethod
    def _unpickle(file):
        with open(file, 'rb') as fo:
            ret_val = pickle.load(fo, encoding='bytes')
        return ret_val

    def _download_cifar10(self):
        cifar10_loading_script = "wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        os.system(cifar10_loading_script)

        cifar10_extract = "tar -xvzf cifar-10-python.tar.gz"
        os.system(cifar10_extract)

    def load_cifar10(self):
        # self._download_cifar10()
        for i in range(1, 6):
            d = self._unpickle(f"cifar-10-batches-py/data_batch_{i}")
        print(d.keys())


if __name__ == "__main__":
    loader = DataLoader()
    loader.load_cifar10()
