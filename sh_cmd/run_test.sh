export CUDA_VISIBLE_DEVICES=2,3
for i in $(seq 1 500); do
    torchrun --nproc_per_node 2 --master_port 1235 run.py \
    --do_not_save_results \
    --cfg experiment/llama2_baseline.cfg \
    --scheme w_def \
    --split test \
    --max_new_tokens 256 \
    --report_to none \
    --output_dir=./results/test/ \
    --overwrite_output_dir \
    --use_dataset_cache \
    --do_predict \
    --per_device_eval_batch_size 8 \
    --remove_unused_columns False \
    --seed 42
done