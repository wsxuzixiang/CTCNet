# CTCNet: A CNN-Transformer Cooperation Network for Face Image Super-Resolution in PyTorch 

[A CNN-Transformer Cooperation Network for Face Image Super-Resolution](https://arxiv.org/abs/2204.08696v2)  
[**Guangwei Gao**](https://guangweigao.github.io/), [Zixiang Xu](https://github.com/wsxuzixiang)

![example result](https://github.com/wsxuzixiang/CTCNet/blob/main/Network.png)

## Comparisons for ✖️8 SR on the CelebA test set.
![example result](https://github.com/wsxuzixiang/CTCNet/blob/main/img/compare_CelebA_0213.png)
![example result](https://github.com/wsxuzixiang/CTCNet/blob/main/img/Snipaste_2023-05-09_22-20-55.png)


## Installation and Requirements 

Clone this repository
```
git clone https://github.com/IVIPLab/CTCNet
cd CTCNet
```

I have tested the codes on
-install required packages by `pip install -r requirements.txt`  


### Test with Pretrained Models

We provide example test commands in script `test.sh` for both CTCNet. Two models with difference configurations are provided for each of them, refer to [section below](#differences-with-the-paper) to see the differences. Here are some test tips:

- CTCNet upsample a 16x16 bicubic downsampled face image to 128x128, and there is **no need to align the LR face**.   
- Please specify test input directory with `--dataroot` option.  
- Please specify save path with `--save_as_dir`, otherwise the results will be saved to predefined directory `results/exp_name/test_latest`.  

### Train the Model

The commands used to train the released models are provided in script `train.sh`. Here are some train tips:

- You should download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train CTCNet and CTCGAN respectively. Please change the `--dataroot` to the path where your training images are stored.  
- To train CTCNet, we simply crop out faces from CelebA without pre-alignment, because for ultra low resolution face SR, it is difficult to pre-align the LR images.  
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `check_points/log_archive`, and the weight directory will only store weight history of latest experiment with the same name.  
- `--gpus` specify number of GPUs used to train. The script will use GPUs with more available memory first. To specify the GPU index, uncomment the `export CUDA_VISIBLE_DEVICES=` 

### Pretrained models

The **pretrained models** and **test results** can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sJs2JYqddSk1o4hksOrO2Fk2ciRelXUQ) .

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
