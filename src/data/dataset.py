from datasets import Dataset

def sft_formatter(example):
    return f"### Question: {example['prompt']}\n ### Answer: {example['ground_truth']}"
    

class DATASET:
    def __init__(self, train, test):
        self.train = Dataset.from_pandas(train)
        self.test = Dataset.from_pandas(test)

    