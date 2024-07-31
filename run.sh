torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.125 --timestep 56 >>timestep56.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.125 --timestep 48 >>timestep48.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.15 --timestep 40 >>timestep40.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.2 --timestep 32 >>timestep32.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.25 --timestep 24 >>timestep24.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.3 --timestep 16 >>timestep16.txt

torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model relu_swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.1 --timestep 56 >>relu_timestep56.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model relu_swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.1 --timestep 48 >>relu_timestep48.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model relu_swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.125 --timestep 40 >>relu_timestep40.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model relu_swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.15 --timestep 32 >>relu_timestep32.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model relu_swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.2 --timestep 24 >>relu_timestep24.txt
torchrun --nproc_per_node 3 inference_swin.py /workspace/dataset/imagenet --model relu_swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.25 --timestep 16 >>relu_timestep16.txt
