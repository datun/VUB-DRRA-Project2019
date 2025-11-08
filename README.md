# VUB - Data Representation, Reduction and Analysis Project
------

# Image Super Resolution - Using ESPCN
This repository contains the implementation of ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/abs/1609.05158).  
The coding of this is done in Python by using PyTorch, [example provided by pytorch](https://github.com/pytorch/examples/tree/master/super_resolution) forms the base of this code.

## Requirements
## Libraries
- PyTorch
- Numpy
- Pillow

### Files

| Training Files                                         | Upscaling Files                        |
| ------------------------------------------------------ | -------------------------------------- |
| At least one image dataset contained within a zip file | At least one image and a trained model |
| Optional: Verification/Test image dataset              |                                        |

## How to Train 

```sh
python train.py --upscale 3 --trainBatchSize 4 --validBatchSize 100 
--nEpochs 30 --lr 0.001 --cuda --cropSize 420 --func leaky
```

Above is an example code to be passed to command line to start training.
After that user will be prompted for:
- Training dataset location (including the zip file)  
It accepts relative paths or direct paths (ex: `dataset\training_imageset.zip`)
- Whether the images are contained within a folder or not (1/0)
  - If yes: ask for folder structure until images are found  
    (ex: `training_folder\another_folder`)
- Will repeat process above for validation datasets.  
In case of one dataset, training set can be used as validation but it will yield  
positively biased PSNR results.

The results will be saved in a folder of `logs_scale_#UPSCALE#_crop_#CROPSIZE#`
 
**IMPORTANT:** 
- For next training, rename or move the result folder.    
(Else, logs may be extended or the code will stop due to existing files)
- If the code fails, remove the result folder and extr folder from root directory.  
(existence of extr folder will yield an error, stopping the code)

## Upscaling Images
There are two files that can be used to upscale images:  
**super_res.py** and **batch_SR-PSNR_calculation.py**

### Super_res.py commands

```sh
python super_res.py --input LowRes-Image.png 
--model processed\logs-models_scale_%i\epoch_%i_model.pth 
--output SuperRes-Image.png --cuda
```

The arguments are straightforward, pass the path of input image, model and where the output is desired.

There is an optional parameter `--compare HighRes-Image.png` where it will print out the PSNR of upscaled image compared to the given High Resolution Image.

### Batch_SR-PSNR_calculation.py

```sh
python batch_SR-PSNR_calculation.py
--model processed\logs-models_scale_%i\epoch_%i_model.pth
--cuda
```

Unlike super_res.py file, it only takes the model as argument.
- Then it will first ask for image set to be upscaled
- And high resolution counterparts of the low res images.

Resulting SR images are saved in `SR_results` folder with the PSNR result logs.

### Batch_bicubic_PSNR_calculation.py
The way this works is identical to batch_SR-PSNR_calculation.py  
Only difference is that while running in terminal it only expects

`--cuda` as an argument, no models required.  
It will ask for low-res image sets and high-res image sets so that it will upscale the low-res images by using BICUBIC and calculate the PSNR.

Note that this file will not output the BICUBIC upscaled images, only the PSNR calculations.
