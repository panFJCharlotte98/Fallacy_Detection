for model in gpt gpt4; do
  for seed in 42 0 123 1234 12345; do
    for scheme in w_def wo_def; do
      torchrun --nproc_per_node 1 --master_port 1237 run.py \
        --which_task=argotario,logic,elecdebate,reddit,propaganda,mafalda,covid \
        --cfg experiment/${model}_baseline.cfg \
        --scheme ${scheme} \
        --max_new_tokens 256 \
        --report_to none \
        --output_dir=./results \
        --overwrite_output_dir \
        --do_predict \
        --per_device_eval_batch_size 1 \
        --remove_unused_columns False \
        --seed ${seed}
    done
  done
done


for model in gpt gpt4; do
  for seed in 42 0 123 1234 12345; do
    for scheme in v1_wo_def v12_wo_def v2_gen_def v3_cot_wo_def v13_wo_def v14_wo_def; do
      torchrun --nproc_per_node 1 --master_port 1236 run.py \
        --cfg experiment/${model}_multiprompt.cfg \
        --scheme ${scheme} \
        --max_new_tokens 640 \
        --report_to none \
        --output_dir=./results \
        --overwrite_output_dir \
        --do_predict \
        --per_device_eval_batch_size 1 \
        --remove_unused_columns False \
        --seed ${seed}
    done
  done
done