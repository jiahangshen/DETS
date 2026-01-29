import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
file_path = 'enthalpy/wudily_cho.csv' 
df = pd.read_csv(file_path)
data = df['ΔfH°(298.15 K)']

# 2. 确定整数边界 (Bin Edges)
# 向下取整最小值，向上取整最大值，确保覆盖所有数据
min_val = np.floor(data.min())
max_val = np.ceil(data.max())

# 生成步长为 1 的整数序列作为 bins
# 例如：[-5, -4, -3, ..., 4, 5]
bins = np.arange(min_val, max_val + 1, 1)

# 3. 统计 [-1, 1] 范围内的数据
in_range_count = data[(data >= -1) & (data <= 1)].count()
total_count = data.count()
percentage = (in_range_count / total_count) * 100

print(f"数据总数: {total_count}")
print(f"[-1, 1] 范围内数量: {in_range_count}")
print(f"占比: {percentage:.2f}%")

# 4. 绘图
plt.figure(figsize=(12, 6))

# 绘制直方图，edgecolor='white' 让柱子之间有明显的白色分隔线
n, bins_out, patches = plt.hist(data, bins=bins, color='#4c72b0', edgecolor='white', alpha=0.8)

# 5. 高亮 [-1, 1] 区域
# 在直方图中，[-1, 1] 对应的是两个整数区间：[-1, 0] 和 [0, 1]
plt.axvspan(-1, 1, color='orange', alpha=0.3, label='Target Range [-1, 1]')
plt.axvline(-1, color='red', linestyle='--', linewidth=1.5)
plt.axvline(1, color='red', linestyle='--', linewidth=1.5)

# 6. 设置 X 轴刻度为整数
plt.xticks(bins)  # 强制显示所有整数刻度
plt.xlim(min_val, max_val) # 限制显示范围

# 添加标签和标题
plt.title(f'Integer Bin Histogram of dGsolv_avg\nRange [-1, 1] Percentage: {percentage:.2f}%', fontsize=14)
plt.xlabel('dGsolv_avg (Integer Intervals)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.legend()

# 保存并显示
plt.tight_layout()
plt.savefig('enthalpy/theo_integer_histogram.png', dpi=300)
plt.show()