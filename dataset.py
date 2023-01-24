from abc import ABC
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_loader import load_data


class KGDataset(Dataset, ABC):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item].tolist()

    @staticmethod
    def collate_fn(batch):
        entity_id = [int(b[0]) for b in batch]
        relation_id = [b[1] for b in batch]
        object_id = [b[2] for b in batch]

        return {
            'entity_id': torch.tensor(entity_id, dtype=torch.int64),
            'relation_id': torch.tensor(relation_id, dtype=torch.int64),
            'object_id': torch.tensor(object_id, dtype=torch.int64),
        }


class RatDataset(Dataset, ABC):
    def __init__(self, data):
        self.data = data

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item].tolist()

    @staticmethod
    def collate_fn(batch):
        user_id = [int(b[0]) for b in batch]
        entity_id = [int(b[1]) for b in batch]
        score = [int(b[2]) for b in batch]

        return {
            'user_id': torch.tensor(user_id, dtype=torch.int64),
            'entity_id': torch.tensor(entity_id, dtype=torch.int64),
            'score': torch.tensor(score, dtype=torch.int64),
        }


def main():
    data = load_data()
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    kg = data[7]
    dataset = KGDataset(kg)


if __name__ == '__main__':
    main()





