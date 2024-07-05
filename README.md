# [DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_DOLCE_A_Model-Based_Probabilistic_Diffusion_Framework_for_Limited-Angle_CT_Reconstruction_ICCV_2023_paper.pdf)

Limited-Angle Computed Tomography (LACT) is a non-destructive evaluation technique used in a variety of applications ranging from security to medicine. The limited angle coverage in LACT is often a dominant source of severe artifacts in the reconstructed images, making it a challenging inverse problem. We present DOLCE, a new deep model-based framework for LACT that uses a conditional diffusion model as an image prior. Diffusion models are a recent class of deep generative models that are relatively easy to train due to their implementation as image denoisers. DOLCE can form high-quality images from severely under-sampled data by integrating data-consistency updates with the sampling updates of a diffusion model, which is conditioned on the transformed limited-angle data. We show through extensive experimentation on several challenging real LACT datasets that, the same pre-trained DOLCE model achieves the SOTA performance on drastically different types of images. Additionally, we show that, unlike standard LACT reconstruction methods, DOLCE naturally enables the quantification of the reconstruction uncertainty by generating multiple samples consistent with the measured data.

[Project page](https://wustl-cig.github.io/dolcewww/).


## Download pre-trained models and CT dataset

We have released [checkpoints & testing images](https://drive.google.com/drive/folders/1p6d2jdHXOI_09svJ8yw6gybDotSE2s8A?usp=sharing) used in the paper. Please extract data into the ./dataset and models into ./model_zoo after downloading. The model was tested successfully on NVIDIA V100,  RTX3090, A6000, and A100.

you can also download with:
```bash
bash download.sh
```

## How to setup the environment
First install pytorch + leap (from source) using [conda](...) by typing

```
bash install_envs.sh
```
The LEAP source in this repository is tested and fully compatible with [DOLCE](https://github.com/wustl-cig/DOLCE). Its parent (UpToDate) version can be found in [LLNL-LEAP](https://github.com/LLNL/LEAP/tree/main).

### Run the Demo

To demonstrate the performance of dolce on LACT, you can run the following,

For security check-in baggage CT (named as, COE)

```
bash evaluation_securityCT.sh 
```

For medical CT (named as, CKC)

```
bash evaluation_medicalCT.sh
```

The results will be stored in the ./results folder.

## Training datasets

Training images can be downloaded from their original sources, [Medical](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61081171) and [Security](https://alert.northeastern.edu/).

## Citing DOLCE

If you find the paper useful in your research, please cite the paper:
```BibTex
@InProceedings{Liu.etal2023,
    title={DOLCE: A Model-based Probabilistic Diffusion 
                    Framework for Limited-Angle {CT} Reconstruction},
    author={Liu, Jiaming and Anirudh, Rushil and Thiagarajan, Jayaraman J. 
        and He, Stewart and Mohan, K. Aditya and Kamilov, Ulugbek S. and Kim, Hyojin},
    Booktitle={Proc. IEEE Int. Conf. Comp. Vis. (ICCV)},
    year={2023},
    note={arXiv:2211.12340}
}
```

## Citing LEAP

Please also cite the following paper to reference LEAP in your publications:  
```BibTex
@inproceedings{kim2023differentiable,
title={Differentiable Forward Projector for X-ray Computed Tomography},
author={Hyojin Kim and Kyle Champley},
booktitle={ICML 2023 Workshop on Differentiable Almost Everything: Differentiable Relaxations, Algorithms, Operators, and Simulators},
year={2023},
url={https://openreview.net/forum?id=dX8khDFGHv}
}
```    

## Authors
Jiaming Liu (jiaming.liu@wustl.edu): main author  
Hyojin Kim (hkim@llnl.gov): LEAP and dataset curation  


## License
DOLCE is distributed under the terms of the MIT license. All new contributions must be made under this license.
See [LICENSE](LICENSE.txt) in this directory for the terms of the license.  
SPDX-License-Identifier: MIT  
LLNL-CODE-854253  


**Acknowedgement**: Our work is built upon [Guided Diffusion](https://github.com/openai/guided-diffusion).
The LEAP source code included in this repository is slightly modified. Refer to the original [LEAP](GitHub.com/LLNL/LEAP) for more details. 

