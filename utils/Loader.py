import cv2

from utils import NameImg


class Loader:

    @staticmethod
    def load_ranks(dirpath: str):
        """Loads rank images from directory specified by filepath. Stores
        them in a list of Train_ranks objects."""

        if dirpath is not None:
            dirpath = dirpath.removesuffix('/')
            dirpath = dirpath + '/'

        train_ranks = []
        i = 0

        for rank_name in ['14', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']:
            train_ranks.append(NameImg())
            train_ranks[i].name = rank_name
            filename = rank_name + '.jpg'
            train_ranks[i].img = cv2.imread(dirpath + filename, cv2.IMREAD_GRAYSCALE)

            i = i + 1

        return train_ranks

    @staticmethod
    def load_suits(dirpath):
        """Loads suit images from directory specified by filepath. Stores
        them in a list of Train_suits objects."""

        if dirpath is not None:
            dirpath = dirpath.removesuffix('/')
            dirpath = dirpath + '/'

        train_suits = []
        i = 0

        for suit_name in ['S', 'D', 'C', 'H']:
            train_suits.append(NameImg())
            train_suits[i].name = suit_name
            filename = suit_name + '.jpg'
            train_suits[i].img = cv2.imread(dirpath + filename, cv2.IMREAD_GRAYSCALE)
            i = i + 1

        return train_suits
