from dataset import RatDataset, KGDataset
import torch
from torch.utils.data import Subset, DataLoader
import random
from torch.optim import Adam
from MRK import MRK
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def train():
    Epoch = 100
    batch_size = 4096
    dim = 8
    low_layer_nums = 1
    high_layer_nums = 1
    kg_lr = 1e-2
    rs_lr = 2e-2
    device = 'cuda'

    rate_dataset = RatDataset()
    kg_dataset = KGDataset()

    train_index = random.sample(list(range(rate_dataset.length)), int(0.6 * rate_dataset.length))
    eval_test_index = list(set(list(range(0, rate_dataset.length))).difference(set(train_index)))
    eval_index = random.sample(eval_test_index, int(0.5 * len(eval_test_index)))
    test_index = list(set(eval_test_index).difference(set(eval_index)))

    train_dataset = Subset(rate_dataset, indices=train_index)
    eval_dataset = Subset(rate_dataset, indices=eval_index)
    test_dataset = Subset(rate_dataset, indices=test_index)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=rate_dataset.collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=rate_dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=rate_dataset.collate_fn
    )

    kg_dataloader = DataLoader(
        kg_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=kg_dataset.collate_fn
    )

    model = MRK(
        dim=dim,
        user_num=len(rate_dataset.user2id),
        item_num=len(rate_dataset.so2id),
        relation_num=len(rate_dataset.relation2id),
        so_num=len(rate_dataset.so2id),
        low_layer_num=low_layer_nums,
        high_layer_num=high_layer_nums
    )
    model.to(device)
    rs_optimizer = Adam(model.parameters(), lr=rs_lr, weight_decay=1e-6)
    kg_optimizer = Adam(model.parameters(), lr=kg_lr, weight_decay=1e-6)

    for epoch in range(1, Epoch + 1):
        for batch in tqdm(train_dataloader, desc=f'[{epoch:02d}/{Epoch:02d}] training rs'):
            model.train()
            user_ids = batch['user_id']
            entity_id = batch['entity_id']
            score = batch['score']

            output = model(
                user_ids=user_ids.to(device),
                item_ids=entity_id.to(device),
                labels=score.to(device),
                s_ids=entity_id.to(device),
            )
            loss = output[2]
            loss.backward()

            rs_optimizer.step()
            rs_optimizer.zero_grad()

        if epoch % 3 == 0:
            for batch in tqdm(kg_dataloader, desc=f'[{epoch:02d}/{Epoch:02d}] training kg'):
                model.train()
                entity_id = batch['entity_id']
                relation_ids = batch['relation_id']
                object_id = batch['object_id']

                output = model(
                    relation_ids=relation_ids.to(device),
                    item_ids=entity_id.to(device),
                    s_ids=entity_id.to(device),
                    o_ids=object_id.to(device),
                    flag=False
                )
                loss = output
                loss.sum().backward()

                kg_optimizer.step()
                kg_optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            pred_list = []
            pred_normalized_list = []
            label_list = []
            for batch in tqdm(test_dataloader, desc=f'[{epoch:2d}/{Epoch:2d}] evaluate rs'):
                user_ids = batch['user_id']
                entity_id = batch['entity_id']
                score = batch['score']

                output = model(
                    user_ids=user_ids.to(device),
                    item_ids=entity_id.to(device),
                    labels=score.to(device),
                    s_ids=entity_id.to(device),
                )
                pred = output[0]
                pred_normalized = output[1]
                pred_list.extend(pred.tolist())
                pred_normalized_list.extend(pred_normalized.tolist())
                label_list.extend(score.tolist())
            auc = roc_auc_score(y_true=label_list, y_score=pred_list)
            auc_normalized = roc_auc_score(y_true=label_list, y_score=pred_normalized_list)

            print(f'auc: {auc}\tauc_normalized: {auc_normalized}')

        torch.cuda.empty_cache()


def main():
    seed = 1234567890
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train()


if __name__ == '__main__':
    main()
