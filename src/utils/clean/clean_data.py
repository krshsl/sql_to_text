'''
author: krishna
'''
summary = "Summarize the SQL query as a one-line question. If too complex, break it into subqueries, summarize them separately, then combine into one question. Create additional examples to improve understanding and refine the solution."
# please ensure summary is always the first line!!!

import pandas as pd
import numpy as np
import json
import os

dataset_dir = 'dataset'
birdDir = os.path.join(dataset_dir, 'bird')
birdT = os.path.join(birdDir, 'train.json')
birdD = os.path.join(birdDir, 'dev.json')
beaverDir = os.path.join(dataset_dir, 'beaver')
beaverDW = os.path.join(beaverDir, 'dev_dw.json')
beaverNW = os.path.join(beaverDir, 'dev_nw.json')
spider = os.path.join(dataset_dir, 'spider')
spiderT = os.path.join(spider, 'train.json')
spiderD = os.path.join(spider, 'dev.json')

output_dir = 'out'
trainf = pd.DataFrame()
testf = pd.DataFrame()

def clean_data(df, query):
    df = df.dropna()
    return df[[query, 'question', 'db_id']].rename(columns={query: 'query'})

def clean_bird(file):
    df = pd.read_json(file)
    return clean_data(df, 'SQL')

def split_dataset(df):
    msk = np.random.rand(len(df)) <= 0.85
    train = df[msk]
    test = df[~msk]
    return train, test

def clean_beaver(file, query):
    return split_dataset(clean_data(pd.read_json(file), query))

def clean_spider(file):
    df = pd.read_json(file)
    df = clean_data(df, 'query')
    return df.sort_values(['query', 'question', 'db_id'], ascending=[True, True, True]).groupby('query').first().reset_index()

def clean_dataset():
    global trainf, testf
    t1 = clean_bird(birdT)
    d1 = clean_bird(birdD)
    t2, d2 = clean_beaver(beaverDW, 'sql')
    t3, d3 = clean_beaver(beaverDW, 'oracle_sql')
    t4, d4 = clean_beaver(beaverNW, 'sql')
    t5 = clean_spider(spiderT)
    d5 = clean_spider(spiderD)
    trainf = pd.concat([t1, t2, t3, t4, t5])
    testf = pd.concat([d1, d2, d3, d4, d5])
    trainf.sample(frac=1).reset_index(drop=True)
    testf.sample(frac=1).reset_index(drop=True)

def gen_dataset():
    clean_dataset()
    trainf.to_json(os.path.join(output_dir, 'train.json'), orient="split", index=False)
    testf.to_json(os.path.join(output_dir, 'test.json'), orient="split", index=False)

def gen_mini():
    global trainf, testf
    trainm = trainf.sample(70)
    testm = testf.sample(30)
    trainm.to_json(os.path.join(output_dir, 'train_mini.json'), orient="split", index=False)
    testm.to_json(os.path.join(output_dir, 'test_mini.json'), orient="split", index=False)

def get_entry(row):
    data = {}
    question = row.question
    query = row.query
    data["messages"] = [{"role": "system", "content": summary}, {"role": "user", "content": query}, {"role": "assistant", "content": question}]
    return json.dumps(data)

def gen_jsonl():
    global trainf, testf

    with open(os.path.join(output_dir, 'train.jsonl'), 'w') as trainj:
        for i, row in trainf.iterrows():
            trainj.write(get_entry(row))
            trainj.write('\n')

    with open(os.path.join(output_dir, 'test.jsonl'), 'w') as testj:
        for i, row in testf.iterrows():
            testj.write(get_entry(row))
            testj.write('\n')

def create_datasets():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gen_dataset()
    gen_mini()
    gen_jsonl()
