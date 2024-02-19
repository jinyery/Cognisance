import matplotlib.pyplot as plt
import numpy as np

format = "svg"
file_pre = "/mnt/c/Users/yjy/OneDrive/桌面/"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 38  # 设置字体大小
plt.rc('text', usetex=True)
plt.figure(figsize=(15, 10))
# 创建示例数据
categories = ["BLSoftmax", "Logit-Adj", "Mixup", "RandAug", "TADE", "RIDE"]
data1 = [
    [0.4507, 0.4519, 0.4226, 0.488, 0.498, 0.5127],
    [0.3722, 0.3714, 0.3523, 0.408, 0.4204, 0.4288],
    [0.7303, 0.7562, 0.756, 0.7729, 0.768, 0.7911],
    [0.6488, 0.6699, 0.6653, 0.6899, 0.6795, 0.7079],
]  # 第一组数据
data2 = [
    [0.4971, 0.5029, 0.5428, 0.5553, 0.5141, 0.5425],
    [0.4161, 0.424, 0.4575, 0.4751, 0.4337, 0.4628],
    [0.763, 0.7727, 0.8049, 0.8103, 0.7908, 0.8111],
    [0.6834, 0.6853, 0.7197, 0.7172, 0.7074, 0.7283],
]  # 第二组数据
tmp = 3
ax = plt.gca()
if tmp == 0:
    ax.set_ylim(0.35, 0.65)
    filename = "improve_imgnet_clt."+format
elif tmp == 1:
    ax.set_ylim(0.35, 0.65)
    filename = "improve_imgnet_glt."+format
elif tmp == 2:
    ax.set_ylim(0.6, 0.9)
    filename = "improve_mscoco_clt."+format
elif tmp == 3:
    ax.set_ylim(0.6, 0.9)
    filename = "improve_mscoco_glt."+format
ax.xaxis.set_tick_params(rotation=18)

# 设置柱形宽度
bar_width = 0.35
bar_interval = 0.03

# 生成 x 坐标位置
x = np.arange(len(categories))

# 创建第一组柱形图
plt.bar(
    x, data1[tmp], width=bar_width, label="Original", color="royalblue", align="center"
)

# 创建第二组柱形图，将 x 坐标右移 bar_width 以实现双排效果
plt.bar(
    x + bar_width + bar_interval,
    data2[tmp],
    width=bar_width,
    label=r'+\textsc{Cognisance}',
    color="deeppink",
    align="center",
)

# 设置 x 轴标签和标题
# plt.xlabel('Categories')
plt.ylabel("F1-Score$^*$")
# plt.title('Double Bar Chart')

# 设置 x 轴刻度标签
plt.xticks(x + bar_width / 2 + bar_interval / 2, categories)

# 添加图例
plt.legend()
plt.savefig(file_pre + filename, format=format, bbox_inches="tight")
# 显示图形
plt.show()
