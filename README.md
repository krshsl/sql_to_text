# SQL to Natural Language Translation

Our project aims to translate an SQL query to an English sentence for someone to understand. We use 3 different datasets (BIRD, Spider, and Beaver) to summarize a given SQL query. We're using Fireworks AI to finetune and train our models. We employ various strategies.

## Overview

Our system takes SQL queries as input and generates natural language explanations that describe what the query is doing. We leverage three benchmark datasets (BIRD, Spider, and Beaver) and fine-tune large language models using Fireworks AI to achieve high-quality translations.

## Datasets

- **BIRD**: A comprehensive dataset of complex SQL queries with natural language descriptions
- **Spider**: Cross-domain context-independent question-SQL pairs
- **Beaver**: A specialized dataset for SQL-to-text generation

## Approach

We employ several strategies to improve translation quality:

1. **Fine-tuning LLMs**: We fine-tune models on Fireworks AI to understand SQL structure and produce coherent explanations
2. **Template-based Generation**: Using structured templates for consistent outputs

## Getting Started
```bash
# run this command to stop tracking changes to prod.env
# api keys should not be pushed, so this will ensure you don't accidently push your changes
git update-index --assume-unchanged prod.env 

# run this command to start tracking again
git update-index --no-assume-unchanged prod.env 
```

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/krshsl/sql_to_text.git
cd sql_to_text

# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```