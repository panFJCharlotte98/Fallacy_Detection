export CUDA_VISIBLE_DEVICES=0
# # Note: if rerun experiments using new seeds, need to remove arg "use_dataset_cache" to re-generate datasets

# #,elecdebate,propaganda
# for seed in 42; do
#   for model in gpt4; do
#     for scheme in wo_def; do
#       torchrun --nproc_per_node 1 --master_port 1236 run.py \
#         --which_task=propaganda \
#         --n_fewshots 2 \
#         --context_window 0 \
#         --cfg experiment/${model}_fewshot.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 256 \
#         --report_to none \
#         --output_dir=./results \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done

for model in gpt; do
  for seed in 42; do
    for scheme in w_logic_def_qf; do
      torchrun --nproc_per_node 1 --master_port 1236 run.py \
        --which_task=mafalda \
        --cfg experiment/${model}_baseline.cfg \
        --scheme ${scheme} \
        --max_new_tokens 256 \
        --report_to none \
        --output_dir=./results/w_logic_def_qf \
        --overwrite_output_dir \
        --do_predict \
        --per_device_eval_batch_size 1 \
        --remove_unused_columns False \
        --seed ${seed}
    done
  done
done

# for model in gpt4; do
#   for seed in 42; do
#     for scheme in w_def_cf wo_def_cf; do
#       torchrun --nproc_per_node 1 --master_port 1236 run.py \
#         --context_window 0 \
#         --which_task=logic \
#         --cfg experiment/${model}_baseline.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 256 \
#         --report_to none \
#         --output_dir=./results \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done

#argotario,logic,elecdebate,reddit,covid,mafalda
# for model in gpt; do
#   for seed in 123; do
#     for scheme in v2_gen_def_qf; do
#       torchrun --nproc_per_node 1 --master_port 1236 run.py \
#         --context_window 0 \
#         --which_task=covid,logic,mafalda \
#         --cfg experiment/${model}_multiprompt.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 1536 \
#         --report_to none \
#         --output_dir=./results/v2_gen_def_qf \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done


# for model in gpt gpt4; do
#   for seed in 42; do
#     for scheme in w_def wo_def; do
#       torchrun --nproc_per_node 1 --master_port 1237 run.py \
#         --which_task=reddit \
#         --cfg experiment/${model}_baseline.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 256 \
#         --report_to none \
#         --output_dir=./results \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done



# for model in gpt; do
#   for seed in 0 123; do
#     for scheme in w_def wo_def; do
#       torchrun --nproc_per_node 1 --master_port 1237 run.py \
#         --which_task=reddit \
#         --cfg experiment/${model}_baseline.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 256 \
#         --report_to none \
#         --output_dir=./results \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done

# for model in gpt; do
#   for seed in 0 123; do
#     for scheme in v1_wo_def v12_wo_def v2_gen_def v3_cot_wo_def v13_wo_def v14_wo_def v21_gen_def v4_wo_def; do
#       torchrun --nproc_per_node 1 --master_port 1237 run.py \
#         --which_task=reddit \
#         --cfg experiment/${model}_multiprompt.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 640 \
#         --report_to none \
#         --output_dir=./results \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done

# ######## remaining ...
# for model in gpt; do
#   for seed in 42; do
#     for scheme in v21_gen_def v4_wo_def; do
#       torchrun --nproc_per_node 1 --master_port 1237 run.py \
#         --which_task=argotario \
#         --cfg experiment/${model}_multiprompt.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 512 \
#         --report_to none \
#         --output_dir=./results \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done

# for model in gpt; do
#   for seed in 0 123; do
#     for scheme in w_def wo_def; do
#       torchrun --nproc_per_node 1 --master_port 1237 run.py \
#         --which_task=logic,elecdebate,propaganda,covid,mafalda \
#         --cfg experiment/${model}_baseline.cfg \
#         --scheme ${scheme} \
#         --max_new_tokens 512 \
#         --report_to none \
#         --output_dir=./results \
#         --overwrite_output_dir \
#         --do_predict \
#         --per_device_eval_batch_size 1 \
#         --remove_unused_columns False \
#         --seed ${seed}
#     done
#   done
# done

# for seed in 0; do
#   for scheme in w_def wo_def; do
#     torchrun --nproc_per_node 1 --master_port 1237 run.py \
#       --which_task covid \
#       --cfg experiment/gpt_baseline.cfg \
#       --scheme ${scheme} \
#       --max_new_tokens 256 \
#       --report_to none \
#       --output_dir=./results \
#       --overwrite_output_dir \
#       --do_predict \
#       --per_device_eval_batch_size 1 \
#       --remove_unused_columns False \
#       --seed ${seed}
#   done
# done

# # #v1_wo_def v12_wo_def v2_gen_def v3_cot_wo_def 
# # #0 123 1234 12345
# for seed in 0; do
#   for scheme in v1_wo_def v12_wo_def v2_gen_def v3_cot_wo_def v13_wo_def v14_wo_def v21_gen_def v4_wo_def; do
#     torchrun --nproc_per_node 1 --master_port 1237 run.py \
#       --which_task covid \
#       --cfg experiment/gpt_multiprompt.cfg \
#       --scheme ${scheme} \
#       --max_new_tokens 512 \
#       --report_to none \
#       --output_dir=./results \
#       --overwrite_output_dir \
#       --do_predict \
#       --per_device_eval_batch_size 1 \
#       --remove_unused_columns False \
#       --seed ${seed}
#   done
# done


# for seed in 42; do
#   for scheme in v1_wo_def v12_wo_def v2_gen_def v3_cot_wo_def v13_wo_def v14_wo_def v21_gen_def v4_wo_def; do
#     torchrun --nproc_per_node 1 --master_port 1237 run.py \
#       --which_task covid \
#       --cfg experiment/gpt4_multiprompt.cfg \
#       --scheme ${scheme} \
#       --max_new_tokens 512 \
#       --report_to none \
#       --output_dir=./results \
#       --overwrite_output_dir \
#       --do_predict \
#       --per_device_eval_batch_size 1 \
#       --remove_unused_columns False \
#       --seed ${seed}
#   done
# done


# for seed in 42; do
#   for scheme in w_def wo_def; do
#     torchrun --nproc_per_node 1 --master_port 1237 run.py \
#       --which_task covid \
#       --cfg experiment/gpt4_baseline.cfg \
#       --scheme ${scheme} \
#       --max_new_tokens 256 \
#       --report_to none \
#       --output_dir=./results \
#       --overwrite_output_dir \
#       --do_predict \
#       --per_device_eval_batch_size 1 \
#       --remove_unused_columns False \
#       --seed ${seed}
#   done
# done

# for scheme in wo_def w_def; do 
#   torchrun --nproc_per_node 2 --master_port 1237 run.py \
#     --scheme ${scheme} \
#     --cfg experiment/gpt4_baseline.cfg \
#     --max_new_tokens 256 \
#     --report_to none \
#     --output_dir=./results \
#     --overwrite_output_dir \
#     --use_dataset_cache \
#     --do_predict \
#     --seed 42
# done

# #v13_wo_def v1_wo_def v12_wo_def 
# for scheme in v1_wo_def v12_wo_def v3_cot_wo_def v2_gen_def; do
#   torchrun --nproc_per_node 1 --master_port 1237 run.py \
#     --scheme ${scheme} \
#     --cfg experiment/gpt4_multiprompt.cfg \
#     --max_new_tokens 512 \
#     --report_to none \
#     --output_dir=./results \
#     --overwrite_output_dir \
#     --use_dataset_cache \
#     --do_predict \
#     --seed 42
# done

# few shot wo_def w_def
# for setting in wo_def; do 
#   torchrun --nproc_per_node 1 --master_port 1234 run.py \
#     --setting ${setting} \
#     --context_window 1 \
#     --cfg experiment/gpt_fewshot.cfg \
#     --max_new_tokens 256 \
#     --report_to none \
#     --output_dir=./results \
#     --overwrite_output_dir \
#     --use_dataset_cache \
#     --do_predict \
#     --seed 42
# done

# for setting in v2_gen_def; do 
#   torchrun --nproc_per_node 1 --master_port 1234 run.py \
#     --setting ${setting} \
#     --context_window 0 \
#     --cfg experiment/gpt_multiprompt.cfg \
#     --max_new_tokens 512 \
#     --report_to none \
#     --output_dir=./results \
#     --overwrite_output_dir \
#     --use_dataset_cache \
#     --do_predict \
#     --seed 42
# done

# # ablation study: context window for propaganda
# for setting in w_def wo_def; do 
#   torchrun --nproc_per_node 1 --master_port 1234 run.py \
#     --setting ${setting} \
#     --context_window 2 \
#     --cfg experiment/gpt_baseline.cfg \
#     --max_new_tokens 256 \
#     --report_to none \
#     --output_dir=./results \
#     --overwrite_output_dir \
#     --use_dataset_cache \
#     --do_predict \
#     --seed 42
# done