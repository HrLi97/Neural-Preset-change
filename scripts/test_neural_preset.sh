CUDA_VISIBLE_DEVICES=1 python main.py \
model.name=neural_styler \
model.ver=v1 \
model.solver=v1 \
mode=test \
data.name=test_data \
data.batch_size=24 \
data.num_workers=32 \
load.ckpt_path=/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/251107_023532_neural_styler_v1/epoch=0029.ckpt # Update this path to match your downloaded checkpoint filename