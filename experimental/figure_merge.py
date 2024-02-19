import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
categories = ["BLSoftmax", "Logit-Adj", "Mixup", "RandAug", "TADE", "RIDE"]
data1 = [
    [0.4507, 0.4519, 0.4226, 0.488, 0.498, 0.5127],
    [0.3722, 0.3714, 0.3523, 0.408, 0.4204, 0.4288],
    [0.7303, 0.7562, 0.756, 0.7729, 0.768, 0.7911],
    [0.6488, 0.6699, 0.6653, 0.6899, 0.6795, 0.7079]
]  # 第一组数据
data2 = [
    [0.4971, 0.5029, 0.5428, 0.5553, 0.5141, 0.5425],
    [0.4161, 0.424, 0.4575, 0.4751, 0.4337, 0.4628],
    [0.763, 0.7727, 0.8049, 0.8103, 0.7908, 0.8111],
    [0.6834, 0.6853, 0.7197, 0.7172, 0.7074, 0.7283]
]  # 第二组数据

# 设置柱形宽度
bar_width = 0.35
bar_interval = 0.05

# 创建一个包含两行两列的子图布局
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
axes[0][0].set_title("CLT on ImageNet-GLT")
axes[0][1].set_title("GLT on ImageNet-GLT")
axes[1][0].set_title("CLT on MSCOCO-GLT")
axes[1][1].set_title("GLT on MSCOCO-GLT")
axes[0, 0].set_ylim(0.35, 0.65)
axes[0, 1].set_ylim(0.35, 0.65)
axes[1, 0].set_ylim(0.6, 0.9)
axes[1, 1].set_ylim(0.6, 0.9)
# 在每个子图中绘制双排柱形图
for i, ax in enumerate(axes.flatten()):
    x = np.arange(len(categories))
    ax.bar(x, data1[i], width=bar_width, label="Original", color="royalblue", align="center")
    ax.bar(
        x + bar_width + bar_interval,
        data2[i],
        width=bar_width,
        label="Improved",
        color="slateblue",
        align="center",
    )
    # ax.set_xlabel('Categories')
    ax.set_ylabel("F1 Score")
    # ax.set_title(f'Subplot {i+1}')
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()
