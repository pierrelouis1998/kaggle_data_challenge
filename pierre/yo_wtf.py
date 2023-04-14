import numpy as np

from kaggle_data_challenge.utils.load import load_my_data


def main():
    train_data, train_labels, _ = load_my_data()
    train_labels = np.array(train_labels)

    print(np.unique(train_labels))
    print(len(train_labels))
    print(np.count_nonzero(train_labels == 0))
    print(np.count_nonzero(train_labels == 1))

if __name__ == '__main__':
    main()