
torchrun --nproc_per_node 1 inference_swin.py /workspace/dataset/dataset/imagenet --model relu_swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best_relu.pth.tar --base 1.15 --timestep 40