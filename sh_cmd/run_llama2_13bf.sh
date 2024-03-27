# #1. Run baseline
# torchrun --nproc_per_node 2 --master_port 1235 run.py \
# --task argotario \
# --model_tag llama2-13bf \
# --report_to none \
# --output_dir=./results \
# --overwrite_output_dir \
# --use_dataset_cache \
# --run_baseline \
# --max_new_tokens 256 \
# --do_predict \
# --per_device_eval_batch_size 8 \
# --remove_unused_columns False \
# --use_argotario_gold \
# --seed 42

# #2. Run single prompt
# torchrun --nproc_per_node 4 --master_port 1235 run.py \
# --task argotario \
# --model_tag llama2-13bf \
# --report_to none \
# --output_dir=./results \
# --overwrite_output_dir \
# --use_dataset_cache \
# --do_predict \
# --per_device_eval_batch_size 8 \
# --max_new_tokens 850 \
# --remove_unused_columns False \
# --use_argotario_gold \
# --seed 42

#3. Run multi prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results \
--overwrite_output_dir \
--use_dataset_cache \
--run_multiprompt \
--do_predict \
--multipt_start_from 0 \
--per_device_eval_batch_size 8 \
--max_new_tokens 256 \
--remove_unused_columns False \
--use_argotario_gold \
--seed 42


#--top_p 0.5 \
# --temperature 0.5 \
# --top_k 30 \
# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42

# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42



# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42


# Run single prompt
torchrun --nproc_per_node 2 --master_port 1235 run.py \
--task argotario \
--model_tag llama2-13bf \
--report_to none \
--output_dir=./results/occupy \
--overwrite_output_dir \
--use_dataset_cache \
--do_predict \
--per_device_eval_batch_size 8 \
--max_new_tokens 512 \
--remove_unused_columns False \
--do_not_save_results \
--seed 42