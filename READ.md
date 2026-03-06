执行流程（从启动到结束）

  1. 读取 CLI 参数并选择 GPU（torch.cuda.set_device(args.gpu)）train.py:299、
     train.py:306。
  2. 设随机种子（PyTorch/NumPy/Python）和 cuDNN 选项 train.py:82。
  3. 生成实验名，创建 experiments/、logs/、csv/ 并写 CSV 表头 train.py:112、
     train.py:120。
  4. 根据 --dataset 选择路径、预处理、评估函数、输入尺寸 train.py:136。
  5. 创建 DataLoader train.py:173。
  6. 创建模型 SACB_Net(inshape, lp_ratio)，并设置 lp_ratio train.py:189。
  7. 训练损失为 NCC_vxm + Grad3d(l2)，权重由 --weights 控制 train.py:204。
  8. 每个 epoch：训练 -> 验证 Dice -> 写 CSV -> 写 TensorBoard -> 存 checkpoint
     train.py:216、train.py:266、train.py:270。

  参数详解（全部）
  | 参数 | 默认值 | 作用 | 影响 |
  |---|---:|---|---|
  | --dataset | ixi | 选择数据分支：ixi/lpba/abd | 决定数据路径、标签预处理、
  Dice函数、img_size |
  | --lp-ratio | 0.15 | 频域低通半径比例 | 可传 1 个值（4尺度共用）或 4 个值（逐
  尺度）train.py:36 |
  | --weights | 1,0.3 | 两项损失权重（图像相似度、形变正则） | 第1项越大更重对
  齐，第2项越大形变更平滑 |
  | --batch-size | 1 | 训练和验证 batch size | 3D 配准显存占用高，通常保持 1 |
  | --lr | 1e-4 | 初始学习率 | 实际按多项式衰减更新 train.py:282 |
  | --max-epoch | 300 | 总 epoch 数 | 训练时长和收敛上限 |
  | --epoch-start | 0 | 起始 epoch | 用于续训时从中间继续 |
  | --cont-training | False | 是否续训 | 开启后会从实验目录中加载“最后一个”权重
  |
  | --resume-epoch | 201 | 续训默认起点 | 仅在 cont_training=True 且
  epoch_start=0 时生效 train.py:133 |
  | --seed | 0 | 全局随机种子 | 控制可复现性（含 DataLoader worker）train.py:94
  |
  | --num-workers | 8 | DataLoader 并行加载进程 | 影响数据吞吐；机器弱时可降到
  2/0 |
  | --base-dir | 环境变量 base_dir 或 D:/ | 数据根目录 | 会拼出 IXI_data/,
  LPBA_data_2/, AbdomenCTCT/ |
  | --gpu | 0 | 使用哪块 GPU | 脚本是 CUDA 强依赖（model.cuda()） |
  | --save-tag | '' | 自定义实验标签 | 空时自动生成 dataset_lp... |
  | --cuda-deterministic | False | 是否启用 cuDNN 确定性 | 开启更可复现，通常稍
  慢 train.py:86 |









把 <DATA_ROOT> 替换成你的数据根目录（里面应有 IXI_data/、LPBA_data_2/、AbdomenCTCT/ 之一）。

  1. IXI 快速冒烟（先确认流程通）

  cd /Users/xushuo/Documents/SACB_Net
  python train.py \
    --dataset ixi \
    --base-dir <DATA_ROOT> \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --max-epoch 3 \
    --batch-size 1 \
    --num-workers 2 \
    --gpu 0 \
    --save-tag ixi_smoke_lp015

  2. IXI 正式训练（基线）

  python train.py \
    --dataset ixi \
    --base-dir <DATA_ROOT> \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --lr 1e-4 \
    --max-epoch 300 \
    --batch-size 1 \
    --num-workers 8 \
    --gpu 0 \
    --save-tag ixi_base_lp015

  3. LPBA 正式训练

  python train.py \
    --dataset lpba \
    --base-dir <DATA_ROOT> \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --lr 1e-4 \
    --max-epoch 300 \
    --batch-size 1 \
    --num-workers 8 \
    --gpu 0 \
    --save-tag lpba_base_lp015

  4. Abdomen CT-CT 正式训练

  python train.py \
    --dataset abd \
    --base-dir <DATA_ROOT> \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --lr 1e-4 \
    --max-epoch 300 \
    --batch-size 1 \
    --num-workers 8 \
    --gpu 0 \
    --save-tag abd_base_lp015

  5. 四尺度 lp_ratio（逐尺度设置）

  python train.py \
    --dataset ixi \
    --base-dir <DATA_ROOT> \
    --lp-ratio 0.12,0.14,0.16,0.18 \
    --weights 1,0.3 \
    --max-epoch 300 \
    --batch-size 1 \
    --num-workers 8 \
    --gpu 0 \
    --save-tag ixi_multiscale_lp

  6. 单值扫参（bash）

  for r in 0.10 0.12 0.15 0.18; do
    python train.py \
      --dataset ixi \
      --base-dir <DATA_ROOT> \
      --lp-ratio $r \
      --weights 1,0.3 \
      --max-epoch 100 \
      --batch-size 1 \
      --num-workers 8 \
      --gpu 0 \
      --save-tag ixi_lp${r}
  done

  7. 续训（同一 save-tag）

  python train.py \
    --dataset ixi \
    --base-dir <DATA_ROOT> \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --max-epoch 300 \
    --batch-size 1 \
    --num-workers 8 \
    --gpu 0 \
    --save-tag ixi_base_lp015 \
    --cont-training \
    --epoch-start 120

  ———






### 1. GPU 选择保护（高优先级）

  位置：train.py:299、train.py:306

  当前风险：torch.cuda.get_device_name() 和 set_device() 在无 GPU 或编号越界时直接异常。

  建议改法：在 if __name__ == '__main__': 里先做完整校验，再决定设备。
  如果你暂时不准备支持 CPU 训练，建议 fail-fast（明确报错并退出），不要“假回退”。

  示例逻辑：

  GPU_num = torch.cuda.device_count()
  if not torch.cuda.is_available() or GPU_num == 0:
      raise RuntimeError("CUDA is not available. This training script currently requires GPU.")

  if args.gpu < 0 or args.gpu >= GPU_num:
      raise ValueError(f"--gpu must be in [0, {GPU_num - 1}], got {args.gpu}")

  for i in range(GPU_num):
      print(f"GPU #{i}: {torch.cuda.get_device_name(i)}")

  torch.cuda.set_device(args.gpu)
  print(f"Currently using: {torch.cuda.get_device_name(args.gpu)}")

  ———

  ### 2. 续训加载更稳健（高优先级）

  位置：train.py:195、train.py:198

  当前风险：目录空会崩；os.listdir()[-1] 不保证是 checkpoint；无法指定恢复文件。

  建议改法：

  1. 增加参数 --resume-path（可选，优先级高于自动查找）。
  2. 自动查找时只匹配 *.pth.tar。
  3. 判空后给清晰错误。
  4. torch.load(..., map_location='cpu') 再 load_state_dict，减少设备耦合。

  建议新增参数：

  parser.add_argument('--resume-path', type=str, default='',
                      help='Optional checkpoint path for resuming training.')

  建议新增函数（示意）：

  from pathlib import Path
  from natsort import natsorted

  def resolve_resume_ckpt(exp_dir, resume_path=''):
      if resume_path:
          p = Path(resume_path)
          if not p.is_file():
              raise FileNotFoundError(f"Resume checkpoint not found: {p}")
          return p
      ckpts = natsorted(Path(exp_dir).glob("*.pth.tar"))
      if not ckpts:
          raise FileNotFoundError(f"No checkpoint found in {exp_dir}")
      return ckpts[-1]

  加载时：

  ckpt_path = resolve_resume_ckpt(exp_dir, args.resume_path)
  ckpt = torch.load(str(ckpt_path), map_location='cpu')
  model.load_state_dict(ckpt['state_dict'])

  ———

  ### 3. 验证 batch>1 的 Dice silent 错误（高优先级）

  位置：train.py:183、utils.py:105、utils.py:122、utils.py:143

  当前风险：dice_abdo/dice_LPBA/dice_val_VOI 都用了 [0, 0, ...]，只评估 batch 第一个样本。

  你有两个选项：

  1. 低风险快修：验证强制 batch_size=1。
     把 val_loader 的 batch_size 固定成 1，并打印提示。
  2. 正规修复：把 3 个 Dice 函数改成 batch-aware。
     把 pred = ...[0,0,...] 改成 pred = ...[:,0,...]，循环 batch 求均值返回。

  如果你想先稳，建议先做选项 1，再做选项 2。

  ———

  ### 4. CSV 被覆盖（中优先级）

  位置：train.py:122

  当前风险：每次启动都 w，续训历史丢失。

  建议改法：

  1. 新训：w 并写表头。
  2. 续训：若文件存在则 a；不存在则 w 并写表头。

  示意：

  csv_exists = os.path.exists(csv_name)
  mode = 'a' if cont_training and csv_exists else 'w'
  with open(csv_name, mode, newline='') as f:
      writer = csv.writer(f)
      if mode == 'w':
          writer.writerow(['Index', 'Dice'])

  ———

  ### 5. 学习率不必每 iteration 重设（中优先级）

  位置：train.py:223

  当前问题：同一 epoch 内每个 iter 设置成同一个值，重复操作。

  建议改法：移到 epoch 循环开头调用一次。

  for epoch in range(epoch_start, max_epoch):
      adjust_learning_rate(optimizer, epoch, max_epoch, lr)
      for data in train_loader:
          ...

  ———

  ### 6. model.eval() 放到验证循环外（中优先级）

  位置：train.py:253

  当前问题：每个验证 batch 都重复 eval()。

  建议改法：

  model.eval()
  with torch.no_grad():
      for data in val_loader:
          ...

  如果你愿意更规范，还可加 reg_model.eval()（虽然它无可学习参数）。

  ———

  ### 7. 路径拼接统一（中优先级）

  位置：train.py:112、train.py:274、train.py:288

  当前问题：字符串拼接依赖尾部 /，跨平台和维护都弱。

  建议改法：统一 pathlib.Path。

  from pathlib import Path

  exp_dir = Path("experiments") / save_dir_name
  log_dir = Path("logs") / save_dir_name
  csv_path = Path("csv") / f"...csv"

  exp_dir.mkdir(parents=True, exist_ok=True)
  log_dir.mkdir(parents=True, exist_ok=True)
  csv_path.parent.mkdir(parents=True, exist_ok=True)

  save_checkpoint 也改成接受 Path，内部 torch.save(state, str(path))。

  ———

  ### 8. 参数有效性校验（中优先级）

  位置：train.py:36、train.py:65

  建议新增 validate_args(args)，在 parse_args() 后立即调用。
  最少检查：

  1. batch_size >= 1
  2. max_epoch > 0
  3. 0 <= epoch_start < max_epoch
  4. num_workers >= 0
  5. lr > 0
  6. weights 两个值都 >= 0
  7. lp_ratio（单值或四值）都 > 0
  8. base_dir 存在且可读

  ———

  ### 9. 未使用的 Logger（低优先级）

  位置：train.py:23

  当前问题：Logger 定义了但未接入，增加噪音。

  建议改法：

  1. 直接删除类（最简单）。
  2. 或真正接入 sys.stdout，并实现 flush 正常透传。
     如果你不需要文件日志，建议删除。

  ———

  ### 10. os.environ['base_dir'] 副作用（低优先级）

  位置：train.py:19、train.py:101

  当前问题：训练脚本修改全局环境变量，不够“纯函数式”。

  建议改法：全程只用局部变量 base_dir = Path(args.base_dir)，不要 setdefault / 回写环境变量。
  比如把：

  os.path.join(os.getenv('base_dir'), 'IXI_data/atlas.pkl')

  改成：

  atlas_dir = base_dir / "IXI_data" / "atlas.pkl"

  ———

  额外建议（可选但很值）

  1. checkpoint 同时保存 optimizer.state_dict() 和 epoch，续训更真实。

