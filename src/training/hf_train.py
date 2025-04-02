from data import DATASET, sft_formatter
from .utils import grpo_reward
from trl import SFTConfig, SFTTrainer, setup_chat_format
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
        self.tokenizer = AutoTokenizer.from_pretrained(payload.value)
        self.model = AutoModelForCausalLM.from_pretrained(payload.value)
        self.output_dir = os.path.join(os.environ["MODEL_DIR"], self.output_dir, payload.value)
        self.trainer = None

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.output_dir)

class GRPO(TRAINER):
    output_dir = "grpo"
    def __init__(self, payload, train, test):
        super().__init__(payload, train, test)
        self.training_args = GRPOConfig(output_dir=self.output_dir, overwrite_output_dir=True)
        self.init_model()
    
    def init_model(self):
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=grpo_reward,
            args=self.training_args,
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.test,
        )


class SFT(TRAINER):
    output_dir = "sft"
    def __init__(self, payload, train, test):
        super().__init__(payload, train, test)
        self.training_args = SFTConfig(output_dir=self.output_dir, overwrite_output_dir=True)
        self.mode, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        self.init_model()
    
    def init_model(self):
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            formatting_func=sft_formatter,
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.test,
        )
