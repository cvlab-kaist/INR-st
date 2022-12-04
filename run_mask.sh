python ./scripts/inference_mask_stylization.py \
    --content_path ./samples/content/bear.jpg \
    --style_path ./samples/style/despair.jpg \
    --mask_path ./samples/mask.jpg \
    --save_dir ./checkpoint/bear_despair \
    --width 256 \
    --start_iter 20000 \
    --device_num 0 \