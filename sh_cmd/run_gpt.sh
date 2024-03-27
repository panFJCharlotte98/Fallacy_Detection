# Run baseline
torchrun --nproc_per_node 1 --master_port 1234 run.py \
--task argotario \
--model_tag gpt-3.5-turbo \
--report_to none \
--run_baseline \
--output_dir=./results \
--overwrite_output_dir \
--use_dataset_cache \
--per_device_eval_batch_size 1 \
--max_new_tokens 128 \
--use_argotario_gold \
--seed 42

# Run single prompt
# torchrun --nproc_per_node 1 --master_port 1234 run.py \
# --task argotario \
# --model_tag gpt-3.5-turbo \
# --report_to none \
# --output_dir=./results \
# --overwrite_output_dir \
# --use_dataset_cache \
# --per_device_eval_batch_size 1 \
# --max_new_tokens 512 \
# --use_argotario_gold \
# --seed 42

#3. Run multi prompt
torchrun --nproc_per_node 1 --master_port 1234 run.py \
--task argotario \
--model_tag gpt-3.5-turbo \
--report_to none \
--output_dir=./results \
--overwrite_output_dir \
--use_dataset_cache \
--run_multiprompt \
--multipt_start_from 0 \
--per_device_eval_batch_size 1 \
--max_new_tokens 256 \
--use_argotario_gold \
--seed 42