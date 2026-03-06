1) 项目整体怎么工作（按执行链路）

  1. 训练入口在 train.py#L46。
  2. 数据由 dataset/*.pkl 读入，训练时返回 (x, y)，验证时返回 (x, y, x_seg,
     y_seg)，见 dataset/datasets.py#L12 和 dataset/datasets.py#L137。
  3. 主网络是 SACB_Net，在 model.py#L52。
  4. 编码器提取 5 层特征，见 model.py#L13。
  5. 多尺度配准从最粗层到细层逐步估计位移并上采样融合，见 model.py#L96。
  6. 训练损失当前只用 NCC_vxm + Grad3d，见
     train.py#L144、losses.py#L176、losses.py#L274。

  2) K-means 在哪里、做了什么

  1. K-means 实现在 SACB1.py#L19 的 KM_GPU。
  2. SACB 初始化时持有 self.km，见 SACB1.py#L105。
  3. 真正调用在 SACB1.py#L158 的 forward：
      - 先把局部邻域特征 x_mean 提取出来。
      - 用 K-means 得到 cluster_idx 和 centroid（SACB1.py#L167-SACB1.py#L168）。
      - 每个 cluster 用 centroid 生成动态卷积核和 bias（SACB1.py#L171-
        SACB1.py#L176）。

  这就是“空间共享卷积核”：同一 cluster 内空间位置共享同一组动态卷积参数。

  ———

  3) 你要改哪些部分（必须改）

  如果目标是“高低频二分”直接替代 K-means，最小改动集中在 3 处：

  1. SACB1.py（必须）
      - 删掉 kmeans_gpu 依赖和 KM_GPU（SACB1.py#L12, SACB1.py#L19）。
      - 在 SACB.forward 里把 cluster_idx, centroid = self.km... 换成：
          - fft.fftn + fftshift 对 x_proj 做 3D 频域分解；
          - 用半径阈值（lp_ratio）构建低频掩码；
          - 得到 spatial_low/spatial_high；
          - 用 high > low 得到二分类 cluster_idx（低频类 / 高频类）；
          - 用两类加权平均特征作为 centroid_low/centroid_high，继续喂给
            get_kernel/get_bias。
      - 把 self.num_k 固定成 2，或保留参数但仅允许 2。
      - 同时建议把 out = torch.zeros(...).cuda() 改成 x.new_zeros(...)
        （SACB1.py#L170），避免设备硬编码。
  2. model.py（通常要改）
      - 目前 SACB_Net 还保留 num_k 和 set_k 语义，见 model.py#L57 与
        model.py#L88。
      - 你有两种策略：
          - 保持兼容：set_k 保留但忽略输入（内部强制 2 类）。
          - 完全清理：删除 num_k/set_k 逻辑并同步训练脚本。
  3. train.py（通常要改）
      - k=7 与 model.set_k(k) 是 K-means 超参入口，见
        train.py#L50、train.py#L118、train.py#L119。
      - 频域版下建议替换成 lp_ratio/cutoff 等参数，并写入 tag/save_dir。

  ———

  4) 你仓库里已有的频域代码能不能直接用

  1. SACB_test.py#L14 已经是“频域替代 K-means”的原型，可直接借鉴思路。
  2. 但它还不能直接替换主流程：
      - 主模型导入的是 SACB1.py，不是 SACB_test.py（model.py#L8）。
      - SACB_test.py 没有 cross_Sim，直接替换导入会断。
      - cluster_idx.flatten() 会把 batch 维打平，bs>1 不安全
        （SACB_test.py#L104）。
  3. spectrum_utils.py#L4 目前是 2D FFT（fft2），你的任务是 3D 配准，不能直接
     用，需要改为 fftn(dim=(-3,-2,-1))。

  ———

  5) 推荐改造路线（稳定版）

  1. 在 SACB1.py 内新增 FrequencyPartition（3D FFT + 低频掩码 + 高低频能量图）。
  2. 在 SACB.forward 中生成 cluster_idx 与两个“频率中心特征”，替换 K-means 产
     物。
  3. 保留动态核 get_kernel/get_bias 结构不变，只替换“如何得到类标签和类中心”。
  4. 先保持原损失不变（NCC + Grad3d），做公平对比。
  5. 训练脚本把 k 改成 lp_ratio，并记录到实验名。
  6. 可选地从 requirements.txt 移除 kmeans_gpu（requirements.txt#L2）。

  ———

  如果你愿意，我可以下一步直接给你做一版可运行改动：在 SACB1.py + model.py +
  train.py 完整替换并保持接口尽量兼容，再附一个最小前向测试脚本。

  改造结果

  1. 在 SACB1.py 新增 FrequencyPartition，用 3D FFT + 低通球形掩码划分高低频主导
     区域。
     关键位置: SACB1.py:15, SACB1.py:43
  2. SACB.forward 已替换 K-means 逻辑：

  - 不再调用 kmeans_gpu。
  - 通过 cluster_idx（0 低频 / 1 高频）做空间分组。
  - 用分组后的 x_mean 计算两个“频率中心特征”（对应原先 centroid 作用）。
  - 动态核 get_kernel/get_bias 结构保持不变。
    关键位置: SACB1.py:197, SACB1.py:205, SACB1.py:220

  3. 模型入口新增 lp_ratio 并下发到各尺度 SACB。
     关键位置: model.py:53, model.py:72, model.py:85
  4. 保留兼容接口：

  - set_k() 仍保留，但 FFT 版本 SACB.set_num_k() 内部固定为 2 组。
  - 新增 set_lp_ratio()。
    关键位置: model.py:94, model.py:102, SACB1.py:174, SACB1.py:178

  5. 训练脚本已从 k 切到 lp_ratio，并写入实验 tag。
     关键位置: train.py:50, train.py:52, train.py:118
  6. 新增最小前向测试脚本: smoke_forward_test.py

  验证情况

  1. 语法检查通过：python -m py_compile SACB1.py model.py train.py
     smoke_forward_test.py
  2. 运行前向脚本失败：当前环境缺少 torch（ModuleNotFoundError: No module named
     'torch'），因此无法做运行时前向验证。

  说明

  1. 你方案里的第 6 点“移除 kmeans_gpu 依赖”是可选项，我这次没有改 requirements
     .txt，避免影响你仓库里仍可能使用 K-means 的其他文件（如
     SACB2.py/历史实验脚本）。

  如果你要，我下一步可以继续做两件事：

  1. 给 train.py 增加命令行参数（--lp_ratio），便于扫参。
  2. 再加一个不依赖数据集的 pytest/单测文件，专测 SACB 的 shape 与 bs>1 情况。



  你可以按“从快到慢”的 3 层测试来验证这次改造是否正确。

  1) 快速正确性测试（不依赖数据集）

  1. 确认环境有 torch（你当前环境之前报过缺失）

  python -c "import torch; print(torch.__version__)"

  2. 语法检查

  cd /Users/xushuo/Documents/trae_projects/SACB_Net
  python -m py_compile SACB1.py model.py train.py smoke_forward_test.py

  3. 最小前向检查（我已加好脚本）

  python smoke_forward_test.py

  脚本位置: smoke_forward_test.py

  通过标准:

  1. 能跑通，不报 shape/device 错。
  2. x_warped 和 phi 形状合理（phi 应该是 [B, 3, D, H, W]）。
  3. 输出没有 NaN/Inf。

  ———

  2) 模块级行为测试（仍不依赖数据集）
  重点看你改的频域分组是否稳定，位置在 SACB1.py。
  你至少要验证：

  1. FrequencyPartition 输出 shape 是 [B, D*H*W]，且值只在 {0,1}。
  2. SACB.forward 支持 bs=1 和 bs>1。
  3. lp_ratio 改变时，分组结果会变化（不是“参数无效”）。

  ———

  3) 训练级集成测试（依赖数据集）
  脚本入口: train.py

  建议先做一个“短跑”：

  1. max_epoch 临时设成 1~3。
  2. 观察 loss 是否为有限值且有下降趋势。
  3. 看 csv 是否正常写入，模型文件是否正常保存。
  4. 再做 A/B：同种子、同数据、同 epoch，比较旧版（K-means）和新版（FFT）的
     early Dice/损失趋势。

  ———

  你问的两个“为什么”：

  给 train.py 增加命令行参数的意义是什么？

  1. 不改代码就能切实验（--lp_ratio --dataset --max_epoch），减少手工改文件导致
     的误操作。
  2. 可复现实验更容易（命令即配置）。
  3. 扫参方便（shell 脚本直接循环不同 lp_ratio）。
  4. 日志命名自动规范，便于回溯。
     现在你的参数是硬编码在 train.py:50 等位置，每次改都要动源码。

  为什么要加不依赖数据集的单测文件？

  1. 快：秒级反馈，不用等数据加载和完整训练。
  2. 稳：不会受路径、磁盘、数据缺失影响。
  3. 准：可以精准定位模型逻辑错误（shape、广播、NaN、batch 维问题）。
  4. 可自动化：以后可以放进 CI，防回归。
     注意它不是替代训练测试，而是“第一道防线”。

  ———

  如果你愿意，我下一步可以直接给你补上：

  1. train.py 的 argparse 参数化版本。
  2. 一个 tests/test_sacb_fft.py（无数据集依赖，覆盖 bs=1/2、lp_ratio 生效、输出
     有限值）。


已补上两部分：

  1. train.py 参数化改造
  2. 不依赖数据集的单测文件 tests/test_sacb_fft.py

  代码变更

  1. 给训练脚本新增了 argparse 和参数解析函数，支持 --dataset、--lp-
     ratio、--weights、--max-epoch、--batch-size、--num-workers、--base-
     dir、--gpu 等。
     关键位置: train.py:35, train.py:57, train.py:99, train.py:294
  2. --lp-ratio 支持单值和四尺度值（逗号分隔），例如 0.15 或
     0.1,0.12,0.15,0.18。
     关键位置: train.py:35, train.py:60
  3. 新增无数据依赖单测，覆盖：

  - 频域分组输出 shape + 二值性
  - set_lp_ratio() 后 mask 缓存刷新
  - SACB 在 bs=1/2 下前向 shape 与数值有效性
    关键位置: tests/test_sacb_fft.py:25, tests/test_sacb_fft.py:35, tests/
    test_sacb_fft.py:48

  我已执行的验证

  1. python -m py_compile train.py tests/test_sacb_fft.py 通过。
  2. python -m unittest discover -s tests -p 'test_sacb_fft.py' 通过（当前环境缺
     少 torch，3 个测试被自动 skip，这是预期保护行为）。

  你现在可以直接用的命令

  1. 运行单测（不走数据集）

  cd /Users/xushuo/Documents/trae_projects/SACB_Net
  python -m unittest discover -s tests -p 'test_sacb_fft.py'

  2. 快速短训（验证训练流程）

  python train.py \
    --dataset abd \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --max-epoch 3 \
    --batch-size 1 \
    --num-workers 2 \
    --base-dir /your/dataset/root \
    --gpu 0

  3. 四尺度 lp_ratio 实验

  python train.py \
    --dataset abd \
    --lp-ratio 0.10,0.12,0.15,0.18 \
    --weights 1,0.3 \
    --max-epoch 3 \
    --base-dir /your/dataset/root

  如果你希望，我可以下一步再补一项：让 python train.py --help 在“未安装 torch 的
  环境”也能显示帮助（把重依赖 import 延迟到 main 内部）。

    如何检验代码是否正确（建议顺序）：                                                                                   
                                                                                                                       
  1. 先建兼容环境（README 要求 Python 3.9 + pip<24.1）。                                                               
     README.md                                                                                                         
     requirements.txt                                                                                                  
  2. 跑静态语法检查：                                                                                                  
                                                                                                                       
  python -m py_compile SACB1.py model.py train.py smoke_forward_test.py tests\test_sacb_fft.py                         
                                                                                                                       
  3. 跑模块单测（不依赖数据集）：                                                                                      
                                                                                                                       
  python -m unittest discover -s tests -p "test_sacb_fft.py" -v                                                        
                                                                                                                       
  4. 跑前向冒烟（不依赖数据集）：                                                                                      
                                                                                                                       
  python smoke_forward_test.py                                                                                         
                                                                                                                       
  期望：打印 x_warped shape 和 phi shape，且无 NaN/Inf。                                                               
                                                                                                                       
  5. 跑 1~3 epoch 短训练（依赖数据集）：                                                                               
                                                                                                                       
  python train.py --dataset abd --lp-ratio 0.15 --weights 1,0.3 --max-epoch 1 --batch-size 1 --num-workers 0 --base-dir
  <你的数据根目录> --gpu 0                                                                                             
                                                                                                                       
  期望：loss 有限值、生成 csv 记录、experiments/ 下有 checkpoint。                                                     
                                                                                                                       
  注意：当前代码基本按 CUDA 跑，环境里需要 GPU + torch（例如 train.py 和 losses.py 里有显式 .cuda() / to("cuda")）。   
  train.py#L190                                                                                                        
  losses.py#L297   

  可以，下一步就该上数据集测试。建议按这个顺序做：
                                                                                
  1. 先跑最小数据通路（单进程，先排除 WinError 5）                              
     python train.py --dataset ixi --base-dir D:/ --max-epoch 1 --batch-size 1  
     --num-workers 0 --save-tag smoke_nw0                                       
  2. 再验证多进程 DataLoader（你之前的风险点）                                  
     python train.py --dataset ixi --base-dir D:/ --max-epoch 1 --batch-size 1  
     --num-workers 2 --save-tag smoke_nw2                                       
  3. 验证 lp_ratio 在真实数据上确实生效（同 seed，改 ratio）                    
     python train.py --dataset ixi --base-dir D:/ --max-epoch 1 --batch-size 1  
     --num-workers 0 --seed 0 --lp-ratio 0.10 --save-tag lp010                  
     python train.py --dataset ixi --base-dir D:/ --max-epoch 1 --batch-size 1  
     --num-workers 0 --seed 0 --lp-ratio 0.30 --save-tag lp030                  
  4. 检查产物是否正常生成（csv/ logs/ experiments/），且无报错/NaN。            
                                                                                
  如果第 2 步又出现 WinError 5，就说明问题仍在 Windows 多进程权限链路；训练先固 
  定 --num-workers 0 或 1。                                   