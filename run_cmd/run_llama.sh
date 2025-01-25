# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # Note: if rerun experiments using new seeds, need to remove arg "use_dataset_cache" to re-generate datasets
# # For multi-prompt schemes: --cfg experiment/${model}_multiprompt.cfg
# # For single prompt schemes": --cfg experiment/${model}_baseline.cfg

for model in llama3; do
  for seed in 42 0 123 1234 12345; do
    for scheme in w_def wo_def; do
      torchrun --nproc_per_node 4 --master_port 1234 run.py \
        --context_window 0 \
        --seed ${seed} \
        --which_task=argotario,logic,reddit,elecdebate,propaganda,covid,mafalda \
        --temperature 0.6 \
        --cfg experiment/${model}_baseline.cfg \
        --scheme ${scheme} \
        --per_device_eval_batch_size 32 \
        --max_new_tokens 256 \
        --report_to none \
        --output_dir=./results \
        --overwrite_output_dir \
        --do_predict \
        --remove_unused_columns False
    done
  done
done


for model in llama2; do
  for seed in 42 0 123 1234 12345; do
    for scheme in v1_wo_def v12_wo_def v2_gen_def v3_cot_wo_def v13_wo_def v14_wo_def v21_gen_def v4_wo_def; do
      torchrun --nproc_per_node 2 --master_port 1234 run.py \
        --seed ${seed} \
        --which_task=argotario,logic,reddit,elecdebate,propaganda,covid,mafalda \
        --cfg experiment/${model}_multiprompt.cfg \
        --scheme ${scheme} \
        --per_device_eval_batch_size 16 \
        --max_new_tokens 640 \
        --report_to none \
        --output_dir=./results \
        --overwrite_output_dir \
        --do_predict \
        --remove_unused_columns False
    done
  done
done


for model in mistral; do
  for seed in 42 0 123 1234 12345; do
    for scheme in v1_wo_def v12_wo_def v2_gen_def v3_cot_wo_def v13_wo_def v14_wo_def v21_gen_def v4_wo_def; do
      torchrun --nproc_per_node 2 --master_port 1234 run.py \
        --seed ${seed} \
        --which_task=argotario,logic,reddit,elecdebate,propaganda,covid,mafalda \
        --cfg experiment/${model}_multiprompt.cfg \
        --scheme ${scheme} \
        --per_device_eval_batch_size 32 \
        --max_new_tokens 640 \
        --report_to none \
        --output_dir=./results \
        --overwrite_output_dir \
        --do_predict \
        --remove_unused_columns False
    done
  done
done