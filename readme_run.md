## 环境配置
# 设置huggingface环境变量，否则需要手动下载模型
export HF_ENDPOINT=https://hf-mirror.com


## 运行模型
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/ --no_vis

### 训练ir图像
python scripts/train_d435i.py --epochs 500 --batch_size 2 --lr 1e-4 --out_dir ./train_output_ir/

- 训练ir图像，使用EARR
python scripts/train_d435i.py --epochs 500 --batch_size 4 --accum_steps 4 --img_scale 0.3 --low_memory --mixed_precision --out_dir ./train_output_ir/ —use-earr

- 训练ir图像，不使用earr
python scripts/train_d435i.py --epochs 500 --batch_size 4 --accum_steps 4 --img_scale 0.3 --low_memory --mixed_precision --out_dir ./train_output_ir/ --no_earr


