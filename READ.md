  想提高模型性能，优先调这些参数

  你当前代码里最值得调的就是 train.py:44 这些参数，按收益排序：

  1. --lp-ratio（最高优先级）

  - 推荐先试：0.12 / 0.15 / 0.18 / 0.12,0.14,0.16,0.18
  - 这是 FFT 分区的核心控制参数，通常对 Dice 影响最大。

  2. --weights（图像相似 vs 形变正则）

  - 推荐先试：1,0.2 / 1,0.3 / 1,0.4
  - reg 太小易过拟合形变，太大又对不齐，需平衡。

  3. --lr + --max-epoch（训练预算）

  - 推荐先试：lr=1e-4 和 5e-5
  - 如果有时间预算，max-epoch 从 100 提到 200~300 往往比“乱调结构”更稳定。

  4. --seed（稳定性）

  - 每组最好至少跑 seed=0,1,2，看均值和方差，不要只看单次最好点。

  5. --val-batch-size（主要影响验证速度）

  - 这个主要提速，不直接提升精度。显存够可以设成 2/4。

  ———

  给你一条“小规模提效扫参”命令（8 组，先粗搜）
  （脚本在 scripts/sweep_hparams.sh:1）

  conda activate myenv
  cd /root/SACB_Net
  unset OMP_NUM_THREADS
  export OMP_NUM_THREADS=8
  export CUDA_VISIBLE_DEVICES=0

  bash scripts/sweep_hparams.sh \
    --dataset ixi \
    --base-dir /root/autodl-tmp \
    --gpu 0 \
    --max-epoch 80 \
    --batch-size 1 \
    --num-workers 2 \
    --lp-ratios "0.12;0.15;0.18;0.12,0.14,0.16,0.18" \
    --weights-list "1,0.2;1,0.3" \
    --lrs "1e-4" \
    --seeds "0"

  跑完看 logs/sweeps/.../summary.tsv，再把前 2 组组合拉长到 200~300 epoch 做精搜。