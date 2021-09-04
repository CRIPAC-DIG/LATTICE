import array
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


np.random.seed(123)

folder = './sports/'
name = 'Sports_and_Outdoors'
bert_path = './sentence-bert/stsb-roberta-large/'
bert_model = SentenceTransformer(bert_path)
core = 5

if not os.path.exists(folder + '%d-core'%core):
    os.makedirs(folder + '%d-core'%core)


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

print("----------parse metadata----------")
if not os.path.exists(folder + "meta-data/meta.json"):
    with open(folder + "meta-data/meta.json", 'w') as f:
        for l in parse(folder + 'meta-data/' + "meta_%s.json.gz"%(name)):
            f.write(l+'\n')

print("----------parse data----------")
if not os.path.exists(folder + "meta-data/%d-core.json" % core):
    with open(folder + "meta-data/%d-core.json" % core, 'w') as f:
        for l in parse(folder + 'meta-data/' + "reviews_%s_%d.json.gz"%(name, core)):
            f.write(l+'\n')

print("----------load data----------")
jsons = []
for line in open(folder + "meta-data/%d-core.json" % core).readlines():
    jsons.append(json.loads(line))

print("----------Build dict----------")
items = set()
users = set()
for j in jsons:
    items.add(j['asin'])
    users.add(j['reviewerID'])
print("n_items:", len(items), "n_users:", len(users))


item2id = {}
with open(folder + '%d-core/item_list.txt'%core, 'w') as f:
    for i, item in enumerate(items):
        item2id[item] = i
        f.writelines(item+'\t'+str(i)+'\n')

user2id =  {}
with open(folder + '%d-core/user_list.txt'%core, 'w') as f:
    for i, user in enumerate(users):
        user2id[user] = i
        f.writelines(user+'\t'+str(i)+'\n')


ui = defaultdict(list)
for j in jsons:
    u_id = user2id[j['reviewerID']]
    i_id = item2id[j['asin']]
    ui[u_id].append(i_id)
with open(folder + '%d-core/user-item-dict.json'%core, 'w') as f:
    f.write(json.dumps(ui))


print("----------Split Data----------")
train_json = {}
val_json = {}
test_json = {}
for u, items in ui.items():
    if len(items) < 10:
        testval = np.random.choice(len(items), 2, replace=False)
    else:
        testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

    test = testval[:len(testval)//2]
    val = testval[len(testval)//2:]
    train = [i for i in list(range(len(items))) if i not in testval]
    train_json[u] = [items[idx] for idx in train]
    val_json[u] = [items[idx] for idx in val.tolist()]
    test_json[u] = [items[idx] for idx in test.tolist()]

with open(folder + '%d-core/train.json'%core, 'w') as f:
    json.dump(train_json, f)
with open(folder + '%d-core/val.json'%core, 'w') as f:
    json.dump(val_json, f)
with open(folder + '%d-core/test.json'%core, 'w') as f:
    json.dump(test_json, f)


jsons = []
with open(folder + "meta-data/meta.json", 'r') as f:
    for line in f.readlines():
        jsons.append(json.loads(line))

print("----------Text Features----------")
raw_text = {}
for json in jsons:
    if json['asin'] in item2id:
        string = ' '
        if 'categories' in json:
            for cates in json['categories']:
                for cate in cates:
                    string += cate + ' '
        if 'title' in json:
            string += json['title']
        if 'brand' in json:
            string += json['title']
        if 'description' in json:
            string += json['description']
        raw_text[item2id[json['asin']]] = string.replace('\n', ' ')
texts = []
with open(folder + '%d-core/raw_text.txt'%core, 'w') as f:
    for i in range(len(item2id)):
        f.write(raw_text[i] + '\n')
        texts.append(raw_text[i] + '\n')
sentence_embeddings = bert_model.encode(texts)
assert sentence_embeddings.shape[0] == len(item2id)
np.save(folder+'text_feat.npy', sentence_embeddings)


print("----------Image Features----------")
def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10).decode('UTF-8')
        if asin == '': break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()

data = readImageFeatures(folder + 'meta-data/' + "image_features_%s.b" % name)
feats = {}
avg = []
for d in data:
    if d[0] in item2id:
        feats[int(item2id[d[0]])] = d[1]
        avg.append(d[1]) 
avg = np.array(avg).mean(0).tolist()

ret = []
for i in range(len(item2id)):
    if i in feats:
        ret.append(feats[i])
    else:
        ret.append(avg)

assert len(ret) == len(item2id)
np.save(folder+'image_feat.npy', np.array(ret))


