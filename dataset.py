from torch.utils.data import Dataset


class MDDataset(Dataset):
    def __init__(self, ):
        id2entity_path = 'data/movie/item_index2entity_id.txt'
        kg_path = 'data/movie/kg.txt'
        ratings_path = 'data/movie/ratings.dat'

        with open(id2entity_path, 'r', encoding='utf-8') as f:
            entity_id2index = f.readlines()
        self.entity_id2index = {fi.strip().split('\t')[0]: fi.strip().split('\t')[1] for fi in entity_id2index}
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg = f.readlines()

        with open(ratings_path, 'r', encoding='utf-8') as f:
            ratings = f.readlines()

        relation_set = set()
        for kg_ in kg:
            kg_ = kg_.strip()
            kg_ = kg_.split('\t')
            relation_set.add(kg_[1])
        self.relation2id = {rs: i for i, rs in enumerate(list(relation_set))}

        self.kg = []  # (电影 id, 关系 id, 客体 id)
        self.object = set()
        for kg_ in kg:
            kg_ = kg_.strip()
            kg_ = kg_.split('\t')
            entity_id = kg_[0]
            relation_id = self.relation2id[kg_[1]]
            object_id = kg_[1]
            self.object.add(object_id)
            self.kg.append((
                entity_id, relation_id, object_id
            ))
        self.object = list(self.object)

        user_set = set()
        for rating in ratings:
            rating = rating.strip()
            rating = rating.split('::')
            user_set.add(rating[0])
        self.user2id = {user: i for i, user in enumerate(list(user_set))}

        self.ratings = []  # (user_id, entity_id, score)
        for rating in ratings:
            rating = rating.strip()
            rating = rating.split('::')
            user = rating[0]
            entity_id = rating[1]
            score = int(rating[2])
            self.ratings.append((
                self.user2id[user],
                entity_id,
                1 if score >= 5 else 0
            ))

        self.length = len(self.ratings)


def main():
    dataset = MDDataset()


if __name__ == '__main__':
    main()





