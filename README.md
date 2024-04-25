# Reschedule-Diffusion-based-Bokeh-Rendering

[Paper](https://github.com/Loeiii/Reschedule-Diffusion-based-Bokeh-Rendering) | [Code](https://github.com/Loeiii/Reschedule-Diffusion-based-Bokeh-Rendering) | [Supplementary material](https://oct.org.cn/IJCAI/Supplementary-material.html)

## Data

Training data should be stored in the `datasets` folder in the following format:

```
└── train/
    ├── FNumber_2/                // target images
    ├── FNumber_16/               // input images
├── test/                         
└── validation/
```

Additionally, considering that the dataset has a high resolution which cannot be directly used for training, we also provide `data_resize.py` to adjust the input resolution. You can achieve resolution adjustment by running `data_resize.py` in the `datasets` directory.


```
python data_resize.py -p train -o resizetrain
```

## Training

We provide `train.py` for direct training. You can run it using default parameters or configure training by providing a corresponding `.json	` file.

```
python train.py --config=config/defocus.json
```

## Inference

We provide `infer.py` for direct inference. Additionally, we offer the final experimental weight files for inference. You can [download](https://drive.google.com/drive/folders/18Df3BIfd5hVf_WLCtZwqLTHnN5XQiB8S?usp=sharing)and place it in the `checkpoints` folder. By running the `infer.py` file, you can complete the result sampling.

```
python infer.py --config=config/infer.json
```

## Evaluation

Since the output during model inference is resized data, the metrics should be calculated by first interpolating back to the original resolution, and then computing the metrics. We provide interpolation as well as calculations for metrics like PSNR, SSIM, and LPIPS in `test.py`. By selecting the folder generated by the inference, you can compute these metrics.

```
python test.py --path=experiment/filepath
```

## Visual results



## BibTex

```tex

```

