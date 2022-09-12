import warnings
import h5py
import numpy as np
from sklearn.utils import shuffle
import collections

class ArrayDataset:

    def __init__(self, arrays_dict):

        self.arrays = arrays_dict
        self.size = None

        if self.arrays is not None:
            self.compute_size()
        else:
            self.size = 0

    def save_hdf5(self, save_path):
        file = h5py.File(save_path, "w")

        for key, value in self.arrays.items():
            self.create_dataset_hdf5(file, key, value)

        file.close()

    def load_hdf5(self, load_path):

        file = h5py.File(load_path, "r")

        self.arrays = {}

        for key, value in file.items():
            self.arrays[key] = value[:]

        file.close()

        self.compute_size()

    def compute_size(self):
        self.size = len(list(self.arrays.values())[0])

    def oversample(self, classes):

        indices = np.array(list(range(self.size)), dtype=np.int32)
        y = classes.astype(np.int32)
        x, _ = self.oversample_single(indices, y)

        for key in self.arrays.keys():

            self.arrays[key] = self.arrays[key][x]

    def shuffle(self):

        keys = []
        arrays_list = []

        for key, array in self.arrays.items():
            keys.append(key)
            arrays_list.append(array)

        arrays_list = shuffle(*arrays_list)

        for key, array in zip(keys, arrays_list):

            self.arrays[key] = array

    def split(self, other_size):
        
        other_dataset = {}

        for key, array in self.arrays.items():
            other_dataset[key] = array[:other_size]
            self.arrays[key] = array[other_size:]

        return ArrayDataset(other_dataset)

 
    def delete(self, indices):

        for key, array in self.arrays.items():

            self.arrays[key] = np.delete(array, indices, axis=0)

        self.compute_size()

    @staticmethod
    def create_dataset_hdf5(file, name, data):
        return file.create_dataset(name, data=data)

    @classmethod
    def oversample_single(cls, x, y):
        """
        Oversample the data.
        :param x:       Data.
        :param y:       Labels.
        :return:        Oversampled data.
        """

        _, max_samples = cls.get_min_and_max_samples_per_class(y)
        classes = np.unique(y)

        x, y = shuffle(x, y)

        sampled_x = []
        sampled_y = []

        for class_idx in classes:

            mask = y == class_idx

            if np.sum(mask) == max_samples:

                sampled_x.append(x[mask])
                sampled_y.append(y[mask])

            else:

                sampled_x.append(np.random.choice(x[mask], size=max_samples, replace=True))
                sampled_y.append(np.random.choice(y[mask], size=max_samples, replace=True))

        sampled_x = np.concatenate(sampled_x, axis=0)
        sampled_y = np.concatenate(sampled_y, axis=0)

        return shuffle(sampled_x, sampled_y)

    @staticmethod
    def get_min_and_max_samples_per_class(y):
        """
        Get number of samples for the class with the least and the most samples.
        :param y:       Class indexes for each training sample.
        :return:        Minimum and maximum number of samples.
        """

        min_samples = None
        max_samples = None
        num_classes = np.max(y) + 1

        for class_idx in range(num_classes):

            num_samples = np.sum(y == class_idx)

            if min_samples is None or num_samples < min_samples:
                min_samples = num_samples

            if max_samples is None or num_samples > max_samples:
                max_samples = num_samples

        return min_samples, max_samples

    def __getitem__(self, key):

        return self.arrays[key]

    def __setitem__(self, key, value):

        self.arrays[key] = value

    def __contains__(self, key):

        return key in self.arrays


class ListDataset:

    def __init__(self):

        self.columns = collections.defaultdict(list)

    def add(self, key, value):

        self.columns[key].append(value)

    def get_size(self):

        self.test_equal_sizes()

        if len(self.columns.values()) == 0:
            return 0
        else:
            return len(list(self.columns.values())[0])

    def test_equal_sizes(self):

        if len(self.columns.values()) == 0:
            return True

        ref_size = len(list(self.columns.values())[0])

        for val in self.columns.values():
            assert len(val) == ref_size

    def to_array_dataset(self, dtypes):
        
        assert dtypes.keys() == self.columns.keys()
        self.test_equal_sizes()

        arrays_dict = {}

        for key in dtypes.keys():

            arrays_dict[key] = np.array(self.columns[key], dtypes[key])

        return ArrayDataset(arrays_dict)

def count_objects(string):
    # count the number of objects used in a string
    count = 0

    for i in range(len(string) // 2):

        chars = string[i * 2: (i + 1) * 2]

        if chars == "2b":
            count += 2
        elif chars in ["1b", "1l", "1r", "2r"]:
            count += 1
        else:
            raise ValueError("bad")

    return count

