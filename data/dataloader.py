import os
import pickle
import numpy as np


class DataLoader:
    def __init__(self,
                 seed: int = 42,
                 batch_size: int = 32):
        self._seed = seed
        self._batch_size = batch_size

    @staticmethod
    def _unpickle(file: str):
        with open(file, 'rb') as fo:
            ret_val = pickle.load(fo, encoding='bytes')
        return ret_val

    @staticmethod
    def _download_cifar10():
        cifar10_loading_script = "wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        os.system(cifar10_loading_script)

        cifar10_extract = "tar -xvzf cifar-10-python.tar.gz"
        os.system(cifar10_extract)

    def _get_data_batches(self):
        raw_batch = self._unpickle(f"cifar-10-batches-py/data_batch_1")
        batch_data = raw_batch[b'data']
        print(raw_batch.keys())
        batch_labels = raw_batch[b'labels']
        print(batch_labels[0])

        for i in range(2, 6):
            raw_batch = self._unpickle(f"cifar-10-batches-py/data_batch_{i}")

            batch_data = np.vstack((batch_data, raw_batch[b'data']))
            batch_labels = np.hstack((batch_labels, raw_batch[b'labels']))

        data_batches = (batch_data, batch_labels)
        return data_batches

    def _shuffle(self, a: np.array, b: np.array):
        assert len(a) == len(b)
        np.random.seed(self._seed)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def _split_batches(self, data: np.array, labels: np.array):
        data_ret = [data[i:i + self._batch_size] for i in range(0, len(data), self._batch_size)][:-1]
        labels_ret = [labels[i:i + self._batch_size] for i in range(0, len(labels), self._batch_size)][:-1]

        return np.array(data_ret), np.array(labels_ret)

    def load_cifar10(self):
        if not os.path.exists('cifar-10-batches-py'):
            self._download_cifar10()

        data, labels = self._get_data_batches()
        data, labels = self._shuffle(data, labels)
        data, labels = self._split_batches(data, labels)

        return data, labels


if __name__ == "__main__":
    loader = DataLoader(batch_size=256)
    x, y = loader.load_cifar10()
    print(x.shape)
