from data import DATASET, sft_formatter
from .utils import grpo_reward
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import os


class TRAINER:
    output_dir = None
    def __init__(self, payload, train, test):
        self.payload = payload
        self.dataset = DATASET(train, test)
        self.model = AutoModelForCausalLM.from_pretrained(
                        payload.value,
                        )
        self.output_dir = os.path.join(os.environ["MODEL_DIR"], self.output_dir, payload.value)


class GRPO(TRAINER):
    output_dir = "grpo"
    def __init__(self, payload, train, test):
        super().__init__(payload, train, test)
        self.training_args = GRPOConfig(self.output_dir, auto_find_batch_size=True)
        self.init_model()
    
    def init_model(self):
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=grpo_reward,
            args=self.training_args,
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.test,
        )

    def train(self):
        self.trainer.train()


class SFT(TRAINER):
    output_dir = "sft"
    def __init__(self, payload, train, test):
        super().__init__(payload, train, test)
        self.tokenizer = AutoTokenizer.from_pretrained(payload.value)
        self.training_args = SFTConfig(self.output_dir, auto_find_batch_size=True, packing=True)
        self.init_model()
    
    def init_model(self):
        self.collator = DataCollatorForCompletionOnlyLM(response_template=" ### Answer:", tokenizer=self.tokenizer)
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            formatting_func=sft_formatter,
            data_collator=self.collator,
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.test,
        )

    def train(self):
        self.trainer.train()