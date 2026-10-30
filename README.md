# 平面视觉测量：桌面/圆盘分割与螺栓尺寸估计

用四个彩色标记建立「像素 → 毫米平面」的单应，把二维图像中的几何量搬到毫米平面上测量。项目包含两个子任务：

- **平面目标分割与标定**：分割方桌 / 圆盘的完整 ROI，估计 3×3 单应矩阵，输出方桌四角或圆盘投影圆心；
- **螺栓尺寸测量**：分割单个螺栓轮廓，输出总长 `L`（含头部）与杆部直径 `d`（不含头部）。

实现只使用 OpenCV + NumPy 的传统视觉方法，**无模型、无训练、纯 CPU 可运行**。

## 方法

```text
图像
 ├─ 四色标记检测
 ├─ 归一化 DLT 求单应 H
 ├─ 高斯-牛顿精化（四角 / 圆盘半径约束）
 ├─ 桌面 / 圆盘 / 螺栓 ROI 分割
 └─ 毫米平面几何测量
```

1. **标记检测**：按调色板做颜色距离匹配，形态学去噪，连通域按面积/圆度筛选，校验四点构成凸四边形；失配时用 HSV 兜底。
2. **单应估计**：Hartley 归一化 DLT；方桌用四角做额外约束，圆盘用边界点到圆心的径向距离约束，做 Gauss-Newton 精化。
3. **桌面/圆盘分割**：图像边缘中位数估计背景色，Lab 色差 + 最大连通域 + 填洞；方桌用凸包 + TLS 边线求四角，圆盘用 `fitEllipse`。
4. **螺栓分割**：亮板/暗目标双峰阈值自适应；最大连通域 + 填洞；用最小外接矩形/PCA 定主轴，杆部双侧 TLS 平行线测径，投影极差测总长。
5. **失败处理**：标记缺失、掩膜过小、单应退化、`L/d` 非有限或 `d > L` 时输出 `failed` 和原因，不输出看似合法的错值。

## 结果

在 32 张合成验证图（方桌/圆盘各 16，螺栓 16）上：

| 任务 | 指标 | 数值 |
|---|---|---|
| 桌面/圆盘分割 | mean IoU | **0.9993** |
| 平面标定 | 查询点 MAE | **0.185 mm** |
| 螺栓分割 | mean IoU | **0.9991** |
| 螺栓总长 | MAE | **0.083 mm** |
| 杆部直径 | MAE | **0.097 mm** |
| 有效输出 | 32 / 32 | **100%** |

补充结果：

- 真实场景全图桌面/圆盘分割：**17 / 24** 帧成功，失败均记录原因；
- 48 张真实螺栓裁剪图：mean IoU **0.560**（仅轮廓参考，无毫米真值）；
- 1,042 图 / 20,058 实例工业数据：手工颜色规则基线 recall@IoU0.5 **0.279**，用于说明真实场景的瓶颈在实例级语义，而不在几何换算。

> 以上数字来自程序生成的二维几何数据；真实照片没有卡尺真值，不能当作实物毫米精度。

## 运行

```bash
python -m pip install -r requirements.txt

# 桌面/圆盘 + 螺栓
python starter/run.py --data data --split val --out predictions
python starter/run.py --data data --split test --out predictions_test

# 评测（需要有标签的划分）
python tools/evaluate.py --data data --pred predictions --split val --output evaluation_val.json

# 可视化
python starter/viz.py --data data --split val --out figures --lang zh

# 输出结构自检
python starter/check_requirements.py --data data --pred predictions
```

环境：Python 3.10+，`numpy`、`opencv-python`；`Pillow` 只用于中文标注。

### Windows 中文路径

`starter/run.py` 用 `np.fromfile + cv2.imdecode` 读图，支持中文路径；`tools/evaluate.py` 仍用 `cv2.imread`，在中文目录下会读不到掩膜，导致 IoU 记 0。需要时用：

```bash
python tools/evaluate_safe.py --data data --pred predictions --split val --output evaluation_val.json
```

它只替换读图函数，评测公式与 `tools/evaluate.py` 一致。

## 目录

```text
starter/
  run.py                 运行器
  solution.py            solve_task1 / solve_task2
  vision_lib.py          标记检测、单应、分割、测量
  viz.py                 可视化
  real_demo.py           真实全图 / 真实螺栓裁剪
  ext_industrial.py      工业实例分割数据基线
  manual_demo.py         手动点选 + 失败恢复
  rgbd_measure.py        RGB-D 三维测量（合成自测）
  error_cases.py         最大误差样本归因
  check_requirements.py  输出结构/接口自检
tools/
  evaluate.py            评测脚本
  evaluate_safe.py       中文路径安全包装
  check_dataset.py       数据完整性检查
predictions/             验证集输出
predictions_test/        测试集输出
figures/                 合成数据可视化
real_figures/            真实数据结果（公开仓库不含原图）
```

## 公开仓库说明

`real_figures/` 中的真实项目全图、真实螺栓裁剪图和工业数据基线图来自内部数据，授权状态未声明为公共许可，因此**未上传到公开仓库**；目录中只保留说明文件。完整的本地版本包含这些图。

## 局限

- 平面单应只适用于标定平面上的点；物体有高度、立放、倾斜或遮挡时会产生系统偏差，应输出不可测而不是按常见规格补齐。
- 真实照片没有毫米真值，本项目不报告实物精度；真实数据上的分割差距说明需要学习方法解决实例级语义。
- `rgbd_measure.py` 只在合成场景验证，真实 RGB-D 需要相机内参、深度对齐和完整性检查，尚未在真实数据上验证。
- 标记被完全遮挡时没有 3 点备用解；项目假设四个标记完整可见。

## 资料

- OpenCV 文档：`findContours`、`fitEllipse`、`minAreaRect`、`fillPoly(shift)`、`connectedComponentsWithStats`
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*，第 4 章（归一化 DLT）
- 张正友平面标定法（Zhang, 2000, *TPAMI*）
- 旋转卡壳 / 最小面积外接矩形
- 公开补充数据 [Nuts and Bolts](https://universe.roboflow.com/jente-kleinh/nuts-and-bolts)（CC BY 4.0）

更完整的推导、误差分析和逐例归因见 [`REPORT.md`](REPORT.md)；数据说明见 [`DATA.md`](DATA.md)。
