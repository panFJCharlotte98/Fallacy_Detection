# # run T5-3B multi-task 
for tasks in argotario,logic,elecdebate,propaganda argotario,logic,reddit; do
  for optim in adafactor; do
    for seed in 42 0 123; do
      torchrun --nproc_per_node 4 --master_port 1234 run.py \
      --which_task ${tasks} \
      --optim ${optim} \
      --context_window 0 \
      --cfg experiment/T5_3b_finetune_multitask.cfg \
      --num_train_epochs 5 \
      --learning_rate 1e-4 \
      --lr_scheduler_type constant_with_warmup \
      --warmup_ratio 0.1 \
      --gradient_accumulation_steps 32 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --report_to none \
      --output_dir=./results \
      --overwrite_output_dir \
      --do_train \
      --do_eval \
      --do_predict \
      --logging_strategy steps \
      --logging_first_step \
      --logging_steps 5 \
      --evaluation_strategy epoch \
      --load_best_model_at_end \
      --metric_for_best_model avr \
      --greater_is_better True \
      --save_strategy epoch \
      --save_total_limit 1 \
      --save_safetensors False \
      --seed ${seed}
    done
  done
done

# #run T5-3B single-task: adafactor+1024+no context
# # argotario,logic,reddit,elecdebate,propaganda 
for tasks in argotario,logic,elecdebate,propaganda argotario,logic,reddit; do
  for seed in 42 0 123; do
    torchrun --nproc_per_node 4 --master_port 1234 run.py \
    --which_task ${tasks} \
    --optim adafactor \
    --context_window 0 \
    --cfg experiment/T5_3b_finetune_singletask.cfg \
    -num_train_epochs 5 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --report_to none \
    --output_dir=./results \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --logging_strategy steps \
    --logging_first_step \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model avr \
    --greater_is_better True \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_safetensors False \
    --seed ${seed}
  done
done

# # # run OOD T5-3B multi-task ALEP: adafactor+1024+no context for propaganda
torchrun --nproc_per_node 4 --master_port 1234 run.py \
--which_task=reddit,mafalda,covid \
--cfg experiment/T5_3b_finetune_singletask.cfg \
--per_device_eval_batch_size 1 \
--report_to none \
--output_dir=./results/t5-3b/multi-task/GPU-4_seed-1234_optim-OptimizerNames.ADAFACTOR_lr-0.0001_ep-5.0_gas-32_tbs-1_ebs-1_ALEP_20240928213613 \
--load_weights_from ./results/t5-3b/multi-task/GPU-4_seed-1234_optim-OptimizerNames.ADAFACTOR_lr-0.0001_ep-5.0_gas-32_tbs-1_ebs-1_ALEP_20240928213613/checkpoint-198 \
--do_predict \
--save_safetensors False \
--seed 1234

# # # run OOD T5-3B multi-task ALR: adafactor+1024+no context for propaganda
torchrun --nproc_per_node 4 --master_port 1234 run.py \
--which_task=elecdebate,propaganda,mafalda,covid \
--context_window 0 \
--cfg experiment/T5_3b_finetune_singletask.cfg \
--per_device_eval_batch_size 1 \
--report_to none \
--output_dir=./results/t5-3b/multi-task/GPU-4_seed-1234_optim-OptimizerNames.ADAFACTOR_lr-0.0001_ep-5.0_gas-32_tbs-1_ebs-1_ALR_20240929004657 \
--load_weights_from ./results/t5-3b/multi-task/GPU-4_seed-1234_optim-OptimizerNames.ADAFACTOR_lr-0.0001_ep-5.0_gas-32_tbs-1_ebs-1_ALR_20240929004657/checkpoint-180 \
--do_predict \
--save_safetensors False \
--seed 1234
