EVAL_STEP=250
LOGGING_STEP=250

# CUDA_VISIBLE_DEVICES=2
python run_clip.py \
    --output_dir ./models/vit-b-p16-224-roberta-b-specprop-tr100 \
    --model_name_or_path ./initial_models/vit-b-p16-224-roberta-b \
    --train_file ./data/train.json \
    --validation_file ./data/dev.json \
    --img_dir ./clevr_images/normal \
    --cache_dir cache \
    --image_column image_path \
    --caption_column caption \
    --specify_prop_in_caption \
    --freeze_vision_model \
    --remove_unused_columns=False \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --num_train_epochs 5 \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" \
    --weight_decay 0.01 \
    --load_best_model_at_end \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch
# --evaluation_strategy steps \
# --eval_steps $EVAL_STEP \
# --save_steps $EVAL_STEP \
# --logging_strategy steps \
# --logging_steps $LOGGING_STEP \
