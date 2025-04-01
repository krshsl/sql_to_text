from rouge import Rouge
import numpy as np

rouge = Rouge()

def grpo_reward(prompts, completions, ground_truth, **kwargs):
    rewards = []
    
    for prompt, completion, truth in zip(prompts, completions, ground_truth):
        try:
            scores = rouge.get_scores(completion, truth)[0]
            reward = np.mean([scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']])
        except:
            reward = 0.0
        
        rewards.append(reward)
    
    return np.array(rewards)