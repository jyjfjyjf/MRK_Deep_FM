from data_loader import load_data
from dataset import RatDataset, KGDataset
import torch
from torch.utils.data import DataLoader
import random
from torch.optim import AdamW
from MKRAttention import MKRAttention
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
    rs_lr = 1e-2
    device = 'cuda'

    data = load_data()
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    kg = data[7]
    train_dataset = RatDataset(train_data)
    eval_dataset = RatDataset(eval_data)
    test_dataset = RatDataset(test_data)
    kg_dataset = KGDataset(kg)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=eval_dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=test_dataset.collate_fn
    )

    kg_dataloader = DataLoader(
        kg_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=kg_dataset.collate_fn
    )

    model = MKRAttention(
        dim=dim,
        user_num=n_user,
        item_num=n_item,
        relation_num=n_relation,
        so_num=n_entity,
        low_layer_num=low_layer_nums,
        high_layer_num=high_layer_nums
    )
    model.to(device)
    rs_optimizer = AdamW(model.parameters(), lr=rs_lr)
    kg_optimizer = AdamW(model.parameters(), lr=kg_lr)

    for epoch in range(1, Epoch + 1):
        pred_list = []
        label_list = []
        with tqdm(iterable=train_dataloader, desc=f'[{epoch:02d}/{Epoch:02d}] training rs',
                  unit_scale=True) as pbar:
            total_loss = 0
            step = 0
            for batch in pbar:
                model.train()
                step += 1
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

                total_loss += loss.item()

                pbar.set_postfix({'loss': f'{total_loss / step:.5f}'})

                rs_optimizer.step()
                rs_optimizer.zero_grad()

                pred = output[0]
                pred_list.extend(pred.tolist())
                label_list.extend(score.tolist())
            train_auc = roc_auc_score(y_true=label_list, y_score=pred_list)
            predictions = [1 if i >= 0.5 else 0 for i in pred_list]
            train_acc = np.mean(np.equal(predictions, label_list))

        if epoch % 3 == 0:
            with tqdm(iterable=kg_dataloader, desc=f'[{epoch:02d}/{Epoch:02d}] training kg',
                      unit_scale=True) as pbar:
                total_loss = 0
                step = 0
                for batch in pbar:
                    step += 1
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
                    total_loss += loss.sum().item()

                    pbar.set_postfix({'loss': f'{total_loss / step:.5f}'})

                    kg_optimizer.step()
                    kg_optimizer.zero_grad()

        model.eval()
        pred_list = []
        label_list = []
        for batch in tqdm(eval_dataloader, desc=f'[{epoch:02d}/{Epoch:02d}] evaluate rs'):
            user_ids = batch['user_id']
            entity_id = batch['entity_id']
            score = batch['score']

            output = model(
                user_ids=user_ids.to(device),
                item_ids=entity_id.to(device),
                labels=score.to(device),
                s_ids=entity_id.to(device),
            )
            # loss = output[2]
            # loss.backward()
            # rs_optimizer.step()
            # rs_optimizer.zero_grad()

            pred = output[0]
            pred_list.extend(pred.tolist())
            label_list.extend(score.tolist())
        eval_auc = roc_auc_score(y_true=label_list, y_score=pred_list)
        predictions = [1 if i >= 0.5 else 0 for i in pred_list]
        eval_acc = np.mean(np.equal(predictions, label_list))

        model.eval()
        pred_list = []
        label_list = []
        for batch in tqdm(test_dataloader, desc=f'[{epoch:02d}/{Epoch:02d}] test rs'):
            user_ids = batch['user_id']
            entity_id = batch['entity_id']
            score = batch['score']

            output = model(
                user_ids=user_ids.to(device),
                item_ids=entity_id.to(device),
                labels=score.to(device),
                s_ids=entity_id.to(device),
            )
            # loss = output[2]
            # loss.backward()
            # rs_optimizer.step()
            # rs_optimizer.zero_grad()

            pred = output[0]
            pred_list.extend(pred.tolist())
            label_list.extend(score.tolist())

        pred_list = torch.sigmoid(torch.tensor(pred_list)).tolist()
        test_auc = roc_auc_score(y_true=label_list, y_score=pred_list)
        predictions = [1 if i >= 0.5 else 0 for i in pred_list]
        test_acc = np.mean(np.equal(predictions, label_list))

        print(f'train auc: {train_auc:.4f}\ttrain acc: {train_acc:.4f}\t'
              f'eval auc: {eval_auc:.4f}\teval acc: {eval_acc:.4f}\t'
              f'test auc: {round(test_auc, 3)}\ttest acc: {round(test_acc, 3)}')

        torch.cuda.empty_cache()


def main():
    seed = 3047
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train()


if __name__ == '__main__':
    main()
