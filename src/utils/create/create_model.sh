model=$1
epochs=$2
lora_rank=$3
context_length=$4
train=$5
test=$6
output_model=$7


firectl create sftj \
    --base-model $model \
    --epochs $epochs \
    --lora-rank $lora_rank \
    --max-context-length $context_length \
    --dataset $train \
    --evaluation-dataset $test \
    --output-model $output_model \
    --turbo \
    --job-id $output_model
