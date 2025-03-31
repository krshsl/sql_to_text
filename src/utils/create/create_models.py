'''
author: krishna
'''
import os
from time import sleep
from clean.clean_data import create_datasets

# models = ['accounts/fireworks/models/llama-v3-8b-instruct', 'accounts/fireworks/models/llama-v3p2-3b-instruct', 'accounts/fireworks/models/llama-v3p2-1b-instruct']

# 3b/1b models require separate deployment which makes it more expensive
models = ['accounts/fireworks/models/llama-v3-8b-instruct']
lora_rank = [8, 32] # using default and one layer higher than default - powers of 2
epochs=[2]
context_length=[8192, 16384] # using default context length at the moment
summaries = [
    "Summarize this SQL query as a question in one line.",
    "Summarize the SQL query as a one-line question. If too complex, break it into subqueries, summarize them separately, then combine into one question. Create additional examples to improve understanding and refine the solution."
]
curr_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(curr_dir, 'dataset')
clean_data = os.path.join(dataset_dir, 'clean_data.py')
train_json = os.path.join('out', 'train.jsonl')
test_json = os.path.join('out', 'test.jsonl')
fine_tune = os.path.join(curr_dir, 'fine_tune.csv')
create_model = os.path.join(curr_dir, 'create_model.sh')

def create_summaries():
    iter = 0
    for summary in summaries:
        file = open(clean_data, 'r')
        lines = file.readlines()
        file.close()
        lines[0] = f"summary = \"{summary}\"\n"
        file = open(clean_data, 'w')
        file.writelines(lines)
        file.close()
        create_datasets()
        os.system(f"firectl remove dataset train{iter}")
        os.system(f"firectl remove dataset test{iter}")
        os.system(f"firectl create dataset train{iter} {train_json}")
        os.system(f"firectl create dataset test{iter} {test_json}")
        iter += 1

def create_models(skip):
    o_iter = 0
    track = 0
    file = open(fine_tune, 'w')
    line = "output,model,epoch,rank,context,train,test"
    file.write(line + "\n")
    print(line)
    for model in models:
        for rank in lora_rank:
            for epoch in epochs:
                for context in context_length:
                    for t_iter in range(len(summaries)):
                        line = f"ads{o_iter},{model},{epoch},{rank},{context},train{t_iter},test{t_iter}"
                        print(line)
                        os.system(f"firectl remove sftj ads{o_iter}")
                        os.system(f"firectl remove model ads{o_iter}")

                        if o_iter > skip:
                            os.system(f"{create_model} {model} {epoch} {rank} {context} train{t_iter} test{t_iter} ads{o_iter}")

                        file.write(line + "\n")
                        o_iter += 1
                        track += 1

                    if track == 2:
                        sleep(1800) # wait 30 minutes
    file.close()

def create():
    skip = -1 # skip some models since not all can be run at once
    # create_summaries()
    print("Start", skip)
    create_models(skip)
    print("End")
