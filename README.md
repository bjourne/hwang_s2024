# All-Spikeformer: An End-to-End Spike-based Vision Transformer with a Winner-Oriented Spike Shift for Softmax Operation
## What is?
This is simple test code for All-Spikeformer which submit to ICML 2024.
The code is designed by PyTorch Image Models(TIMM) and SpikingJelly framework.
Beacuse of limitation of supplement and annonymity, one of pre-trained ANN(swin-tiny without ReLU) is uploaded on below google drive. 
* new annonymous google account created for sharing (account name is Allspikeformer)

link for pretrained ANN: https://drive.google.com/file/d/1SsV4KjJdISWiII378TgArzgT0ZQ-KixF/view?pli=1

In the case paper is accepted, we will provide online resource.

In this code, you need to download pre-trained model on google drive.
First, the code will run ANN(swin_tiny_patch4_window7_224) for scaling threshold(or weight normalization) and searching base.
Then, pre-trained parameter will converted to SNN.(main_distributed.py)


# How to RUN
Run
```
torchrun --nproc_per_node "num_of_gpu" main_distributed.py "data_path of ImageNet" --model swin_tiny_patch4_window7_224 --batch-size "batch_size" --resume model_best.pth.tar --base "base B" --timestep "number of timestep"
```
You can see result(accuracy/energy) as "output.log"

Hyper Parameter:
1. Number of Timestep (-t)   (Default: 40)
2. Base B (1,2] *if B=2, it is binary coding (Default: 1.16)


For example, run
```
torchrun --nproc_per_node 4 main_distributed.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.16 --timestep 40
```


 
