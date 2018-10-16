import numpy as np
import random
from sklearn import metrics
from collections import defaultdict

import torch
from torch import nn
from torch.autograd import Variable

"""
Load Data
"""
def load_data():
    data_path = "movie/processed_ratings.dat"
    user_ratings = defaultdict(set)
    max_uid = -1
    max_iid = -1
    with open(data_path, 'r', encoding='utf16') as f:
        for line in f:
            linetuple = line.strip().split("::")
            u = int(linetuple[0])
            i = int(linetuple[1])
            user_ratings[u].add(i)
            max_uid = max(u, max_uid)
            max_iid = max(i, max_iid)

    return max_uid, max_iid, user_ratings


"""
Arguments
"""
class Args(object):
    def __init__(self,
                 learning_rate=0.025,
                 batch_size=50,
                 dimension=10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dimension = dimension


"""
Train / Test dataset
"""
def generate_test(user_ratings):

    user_test = dict()
    for u, item_set in user_ratings.items():
        user_test[u] = random.sample(item_set, 1)[0]
    return user_test

def train_batch(user_ratings, test_ratings, num_item, batch_size):

    # Uniform sampling train data (user, item_rated(pos), item_not_rated)(neg)).
    s = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        # Only the rating values not found in the test data are sampled for train data.
        while i == test_ratings[u]:
            i = random.sample(user_ratings[u], 1)[0]
        # Only the values that not rated by user.
        j = random.randint(1, num_item)
        while j in user_ratings[u]:
            j = random.randint(1, num_item)
        s.append([u, i, j]) # Train data sample
    train = np.array(s)

    return train


def test_batch(user_ratings, test_ratings, num_item):

    for u in user_ratings.keys():
        s = []
        neg_item_list = []
        i = test_ratings[u]
        cnt = 0
        while cnt < 100:
            j = random.choice(range(1, num_item + 1))
            if j not in neg_item_list and j not in user_ratings[u]:
                s.append([u, i, j])
                neg_item_list.append(j)
                cnt += 1
        yield np.array(s), [u, i, neg_item_list]


"""
Bayesian Personalized Ranking
"""
class BPR(nn.Module):
    def __init__(self, num_user, num_item, hidden_size):
        super(BPR, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.hidden_size = hidden_size

        self.user_embedding = nn.Embedding(num_user, hidden_size)
        self.item_embedding = nn.Embedding(num_item, hidden_size)
        self.item_bias = nn.Parameter(torch.zeros(num_item+1))
        self.sigmoid = nn.Sigmoid()


    def Embedding(num_embeddings, embedding_dim, std=0.1):
        emb = nn.Embedding(num_embeddings+1, embedding_dim)
        emb.weight.data.normal_(0, std)
        emb.weight.grad

        return emb

    def forward(self, input):
        u = input[:,0]
        i = input[:,1]
        j = input[:,2]

        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)
        i_b = self.item_bias[i]
        j_emb = self.item_embedding(j)
        j_b = self.item_bias[j]

        # Matrix Factorization predict: u_i > u_j
        x = i_b - j_b + torch.sum(torch.mul(u_emb, (i_emb - j_emb)), 1)

        l2_norm = torch.norm(torch.cat([u_emb, i_emb, j_emb]), 2)
        l2_norm = torch.pow(l2_norm, 2)

        loss = 0.0001 * l2_norm - torch.mean(torch.log(self.sigmoid(x)))

        return loss

    def evalu(self, input):
        u = input[:,0]
        i = input[:,1]
        j = input[:,2]

        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)
        i_b = self.item_bias[i]
        j_emb = self.item_embedding(j)
        j_b = self.item_bias[j]

        # Matrix Factorization predict: u_i > u_j
        x = i_b - j_b + torch.sum(torch.mul(u_emb, (i_emb - j_emb)), 1)


        pred = torch.max(x,torch.zeros(1).cuda())
        mf_auc = torch.mean(pred)

        y_pred = torch.cat([torch.sum(torch.mul(u_emb, i_emb), 1) + i_b, torch.sum(torch.mul(u_emb, j_emb), 1) + j_b], dim=0)
        y_true = torch.cat([torch.ones(100), torch.zeros(100)], dim=0)

        return y_pred, y_true


def ToTensor(data):
    return torch.LongTensor(data).cuda()


def train(train_data, model, opt):
    loss = model(ToTensor(train_data))
    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss

if __name__ == '__main__':
    import sys

    args = Args()

    num_user, num_item, user_ratings = load_data()

    bpr = BPR(num_user, num_item, args.dimension).cuda()

    test_ratings = generate_test(user_ratings)

    optimizer = torch.optim.Adagrad(bpr.parameters(), lr=args.learning_rate)

    for epoch in range(1, 21):
        batch_loss = 0
        for k in range(1, 10000):
            train_data = train_batch(user_ratings, test_ratings, num_item, args.batch_size)
            loss = train(train_data, bpr, optimizer)
            batch_loss += loss

        print("epoch: %d"%epoch)
        print("loss: ",batch_loss.item()/k)
        auc_sum = 0
        for uij, uij_list in test_batch(user_ratings, test_ratings, num_item):
            y_pred, y_true = bpr.evalu(ToTensor(uij))
            auc = metrics.roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().detach().numpy())
            auc_sum += auc
        print("auc: ", auc_sum.item()/num_user)
