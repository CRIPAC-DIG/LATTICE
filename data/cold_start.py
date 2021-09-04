import json
import numpy as np
import os
from collections import defaultdict


np.random.seed(123)

folder = './sports/'
core = 5


file = open(folder + "meta-data/%d-core.json"%core)
jsons = []
for line in file.readlines():
    jsons.append(json.loads(line))

ui = json.load(open(folder + "%d-core/user-item-dict.json"%core))

iu = defaultdict(list)
for u, items in ui.items():
    for i in items:
        iu[i].append(int(u))

testval = np.random.choice(len(iu), int(0.2*len(iu)), replace=False).tolist()
test = testval[:len(testval)//2]
val = testval[len(testval)//2:]

train_ui = {}
test_ui = {}
val_ui = {}
for u, items in ui.items():
    train_ui[u] = [i for i in items if i not in testval]
    val_ui[u] = [i for i in items if i in val]
    test_ui[u] = [i for i in items if i in test]

if not os.path.exists(folder+'0-core'):
    os.mkdir(folder+'0-core')

json.dump(val_ui, open(folder+'0-core/val.json', 'w'))
json.dump(test_ui, open(folder+'0-core/test.json', 'w'))
json.dump(train_ui, open(folder+'0-core/train.json', 'w'))
