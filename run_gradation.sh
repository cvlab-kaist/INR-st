python ./scripts/inference_gradation_seperation.py \
        --content_path ./samples/content/bear.jpg \
        --style_path ./samples/style/despair.jpg \
        --save_dir ./checkpoint/bear_despair \
        --width 256 \
        --start_iter 20000 \
        --device_num 0 \
        --mode gradation \