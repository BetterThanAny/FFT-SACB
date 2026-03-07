# SACB-Net: Spatial-awareness Convolutions for Medical Image Registration
The official implementation of SACB-Net [![CVPR](https://img.shields.io/badge/CVPR2025-68BC71.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_SACB-Net_Spatial-awareness_Convolutions_for_Medical_Image_Registration_CVPR_2025_paper.html)  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.19592) 

## 环境配置
```
#pip < 24.1
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```
上面的配置其实有点问题, 后续再改

每次训练前先做
```
  conda activate myenv
  unset OMP_NUM_THREADS
  export OMP_NUM_THREADS=8
  export CUDA_VISIBLE_DEVICES=0
  python - <<'PY'
  import torch, os
  print("torch:", torch.__version__)
  print("cuda build:", torch.version.cuda)
  print("cuda available:", torch.cuda.is_available())
  print("device_count:", torch.cuda.device_count())
  print("current visible:", os.getenv("CUDA_VISIBLE_DEVICES"))
  if torch.cuda.is_available():
      print("gpu0:", torch.cuda.get_device_name(0))
  PY
```
  通过标准：

  1. cuda available: True
  2. device_count >= 1
  3. 能打印出 GPU 名称。
  
  
## 代码正确性测试

```python -m py_compile train.py model.py utils.py losses.py SACB1.py SACB2.py nn_util.py dataset/datasets.py dataset/trans.py dataset/
  data_utils.py
  python -m unittest discover -s tests -v
  python smoke_forward_test.py
```
  通过标准：

  1. py_compile 无报错。
  2. 单测全通过。
  3. smoke 输出 x_warped shape 和 phi shape，且无异常退出。

## 训练烟测
```
python train.py \
    --dataset ixi \
    --base-dir /root/autodl-tmp \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --max-epoch 1 \
    --batch-size 1 \
    --num-workers 2 \
    --gpu 0 \
    --save-tag sanity_ixi_e1
```


## Dataset
Thanks [@Junyu](https://github.com/junyuchen245) for [the preprocessed IXI data].

[Abdomen CT-CT](https://learn2reg.grand-challenge.org/Datasets/)
[LPBA](https://loni.usc.edu/research/atlases)

## Weights Download
[Google Drive](https://drive.google.com/drive/folders/1XW19iuyCyg3YGmCpLFGGFjdPFi73xxwh?usp=share_link).

## Citation
```bibtex
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Xinxing and Zhang, Tianyang and Lu, Wenqi and Meng, Qingjie and Frangi, Alejandro F. and Duan, Jinming},
    title     = {SACB-Net: Spatial-awareness Convolutions for Medical Image Registration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5227-5237}
}
```

## Acknowledgments
We sincerely acknowledge the [ModeT](https://github.com/ZAX130/SmileCode), [CANNet](https://github.com/Duanyll/CANConv) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects.


