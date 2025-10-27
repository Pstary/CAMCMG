# CUDA_VISIBLE_DEVICES=1 python run_gen.py \
#   --task summarize \
#   --sub_task java \
#   --do_train \
#   --do_eval \
#   --do_test \
#   --do_eval_bleu \
#   --save_last_checkpoints \
#   --model_name_or_path Salesforce/codet5-base \
#   --tokenizer_name Salesforce/codet5-base \
#   --max_source_length 512 \
#   --cache_path /data/DataLACP/rambo/outputs/ramask/cache_dir \
#   --summary_dir /data/DataLACP/rambo/outputs/ramask/summary_dir \
#   --data_dir data_dir \
#   --res_dir /data/DataLACP/rambo/outputs/ramask/res_dir \
#   --output_dir /data/DataLACP/rambo/outputs/ramask/output_dir \
#   --num_train_epochs 1 \
#   --eval_batch_size 1 \
#   --train_batch_size 1 \
#   --max_target_length 30 \
#   --use_ra_mask

gpu_id=1
lang="java"
# optimizer
lr=5e-5
batch_size=10
beam_size=5
epochs=1

# model 
source_length=512
target_length=100

res_dir=/data/DataLACP/rambo/outputs/ramask/res_dir
data_dir=data_dir
model_name=Salesforce/codet5-base 
# ============ Step 1 Training ==============

function train_codet5 () {

output_dir=/data/DataLACP/rambo/outputs/ramask/output_dir
mkdir -p $output_dir
echo "============TRAINING============"
 CUDA_VISIBLE_DEVICES=$gpu_id python run_gen.py  --do_train --do_eval   --do_test \
  --task summarize \
  --sub_task $lang \
  --summary_dir /data/DataLACP/rambo/outputs/ramask/summary_dir \
  --cache_path /data/DataLACP/rambo/outputs/ramask/cache_dir \
  --data_dir $data_dir \
  --res_dir $res_dir \
  --tokenizer_name $model_name \
  --model_name_or_path $model_name \
  --output_dir $output_dir \
  --do_eval_bleu \
  --save_last_checkpoints \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --do_lower_case \
  --beam_size $beam_size --train_batch_size $batch_size \
  --eval_batch_size $batch_size --learning_rate $lr \
  --num_train_epochs $epochs --seed 0 2>&1|tee  $output_dir/train.log
}


# 
train_codet5
