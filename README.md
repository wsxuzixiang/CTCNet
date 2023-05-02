# CTCNet: A CNN-Transformer Cooperation Network for Face Image Super-Resolution in PyTorch 

[A CNN-Transformer Cooperation Network for Face Image Super-Resolution](https://arxiv.org/abs/2204.08696v2)  
[**Guangwei Gao**](https://guangweigao.github.io/), [Zixiang Xu](https://github.com/wsxuzixiang)

![example result](example.gif)

## Installation and Requirements 

Clone this repository
```
git clone https://github.com/IVIPLab/CTCNet
cd CTCNet
```

I have tested the codes on
-install required packages by `pip install -r requirements.txt`  


### Test with Pretrained Models

We provide example test commands in script `test.sh` for both SPARNet and SPARNetHD. Two models with difference configurations are provided for each of them, refer to [section below](#differences-with-the-paper) to see the differences. Here are some test tips:

- SPARNet upsample a 16x16 bicubic downsampled face image to 128x128, and there is **no need to align the LR face**.   
- SPARNetHD enhance a low quality face image and generate high quality 512x512 outputs, and the LQ inputs **should be pre-aligned as FFHQ**.  
- Please specify test input directory with `--dataroot` option.  
- Please specify save path with `--save_as_dir`, otherwise the results will be saved to predefined directory `results/exp_name/test_latest`.  

### Train the Model

The commands used to train the released models are provided in script `train.sh`. Here are some train tips:

- You should download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train CTCNet and CTCGAN respectively. Please change the `--dataroot` to the path where your training images are stored.  
- To train CTCNet, we simply crop out faces from CelebA without pre-alignment, because for ultra low resolution face SR, it is difficult to pre-align the LR images.  
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `check_points/log_archive`, and the weight directory will only store weight history of latest experiment with the same name.  
- `--gpus` specify number of GPUs used to train. The script will use GPUs with more available memory first. To specify the GPU index, uncomment the `export CUDA_VISIBLE_DEVICES=` 


## Differences with the Paper
Since the original codes are messed up, we rewrite the codes and retrain all models. This leads to slightly different results between the released model and those reported in the paper. Besides, we also extend the 2D spatial attention to 3D attention, and release some models with 3D attention. We list all of them below

### CTCNet

We found that extending 2D spatial attention to 3D attention improves the performance a lot. We trained a light model with half parameter number by reducing the number of FAU blocks, denoted as SPARNet-Light-Attn3D. SPARNet-Light-Attn3D shows similar performance with SPARNet. We also released the model for your reference.   

| Model          | DICNet      | SPARNet (in paper) | SPARNet (Released) | SPARNet-Light-Attn3D (Released) |
| -----------    | ----------- | -----------        | -----------        | -----------                     |
| #Params(M)     | 22.8        | 9.86               | 10.52              | 5.24                            |
| PSNR (&#8593;) | 26.73       | 26.97              | **27.43**          | 27.39                           |
| SSIM (&#8593;) | 0.7955      | 0.8026             | **0.8201**         | 0.8189                          |

*All models are trained with CelebA and tested on Helen test set provided by [DICNet](https://github.com/Maclory/Deep-Iterative-Collaboration)*

![example result](example_ultra_facesrx8.png)

### CTCGAN

We also provide network for CTCGAN. For the test dataset, we clean up non-face images, add some extra test images from internet, and obtain a new CelebA-TestN dataset with 1117 images. We test the retrained model on the new dataset and recalculate the FID scores.

Similar as StyleGAN, we use the exponential moving average weight as the final model, which shows slightly better results.

| Model         | SPARNetHD (in paper) | SPARNetHD-Attn2D (Released) | SPARNetHD-Attn3D (Released) |
| -----------   | -----------          | -----------                 | -----------                 |
| FID (&#8595;) | 27.16                | **26.72**                   | 28.42                       |


## Citation
```
@article{gao2023ctcnet,
  title={Ctcnet: a cnn-transformer cooperation network for face image super-resolution},
  author={Gao, Guangwei and Xu, Zixiang and Li, Juncheng and Yang, Jian and Zeng, Tieyong and Qi, Guo-Jun},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgement

The codes are based on [SPARNet](https://github.com/chaofengc/Face-SPARNet). 