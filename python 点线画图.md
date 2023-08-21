# python 点线画图

学习视频的链接：

【Python线图点图--15分钟详解matplotlib.pyplot.plot #011】https://www.bilibili.com/video/BV1jt4y1C72a?vd_source=289a327f5dfc2718f181e3cad64ee708

```python
# 导入相关依赖
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
X = np.arange(0, 12.1, 0.1)
Y = np.sin(X)
Y1 = np.sin(X+1)
Y2 = np.sin(X+2)
Y3 = np.sin(X+3)
Y4 = np.sin(X+4)
Y5 = np.sin(X+5)

"""
0 画布与布局

figure 创建画布
    figsize 设置画布大小
layout 布局
    tight_layout() 紧致布局 当图片大小不够时，会将图中的有效内容截取掉，为了防止这种情况，可以使用紧致布局
"""
fig = plt.figure(figsize=(4,4))
plt.tight_layout()

"""
1 线
"""
# 简写方式 'r--'
# 其中 r 代表颜色 red
# -- 代表线风格为虚线
plt.plot(X, Y, 'r-.', label='line')

# 单独设置颜色和线型
# color 属性设置颜色
# linestyle 设置线型
# linewidth 设置线的粗细
plt.plot(X, Y1, color='lime', linestyle=':', linewidth=2, label='line1')

"""
2 点 

marker 将点从线上标注出来

属性
marker 设置点型
markerfacecolor markeredgecolor 设置点表面颜色和边缘颜色
markersize 设置点大小
markeredgewidth 设置点边缘粗细
"""

plt.plot(X, Y2, color='green', linestyle='--', linewidth=2, marker='o', markerfacecolor='black', markeredgecolor='red', markersize=5, markeredgewidth=0.5, label='line2')

# 散点图画法
# 1 scatter
plt.scatter(X, Y3, label='line3')
# 2 plot
plt.plot(X, Y4, color='green', linestyle='', linewidth=2, marker='o', markerfacecolor='black', markeredgecolor='red', markersize=3, markeredgewidth=0.5, label='line4',zorder=2)

"""
3 坐标轴上字的设置
"""
# 使用 gca() 方法获取坐标轴
ax1 = plt.gca()

# fontname 字体
# fontsize 字体大小
# weight 字体粗细 bold 加粗
# style 字体风格 italic 斜体
ax1.set_title('Big Title', fontname='Arial', fontsize=20, weight= 'bold', style='italic') # 设置图片标题
ax1.set_xlabel('time(UTC)') # 设置 x 轴标签
# 文本支持 lateX
# ax1.set_ylabel('T($^o$C)') # 设置 y 轴标签
ax1.set_ylabel('T($\mu$C)')


"""
4 坐标轴上刻度的设置

xticks, yticks
"""
ax1.set_xticks([0, 2.5, 7, 11]) # 设置显示的刻度
# ax1.set_xticklabels(['J', 'A', 'N', 'E']) # 设置刻度标签，必须和上面一一对应，否则会报错

# 设置刻度上的标点
# axis 确定坐标值: 'x' x 轴, 'y' y 轴, 'both' 双轴
# direction 确定方向: in 朝内, out 朝外
# color 标点颜色
# length width 标点的长度和粗细
ax1.tick_params(axis='x', direction='out', color='blue', length=10, width=2)

# 在 jupyter 中上述代码会打印 plot 信息，将这些代码赋值给一个空变量，使不打印这些 plot 信息，例如
# _ = ax1.tick_params(axis='x', direction='out', color='blue', length=10, width=2)

ax1.set_yticks([0, 0.5, 1])


"""
5 坐标轴上的数
"""
# xlim, ylim: x 轴和 y 轴数据范围
ax1.set_xlim([0, 12])

# xscale, yscale
# 将 y 轴改为指数坐标轴
ax1.set_yscale('log')

# 双轴
# ax2 与 ax1 共用 x 轴, 也就是多了一个 y 轴
ax2 = ax1.twinx()
# ax3 与 ax1 共用 y 轴, 也就是多了一个 x 轴
ax3 = ax1.twiny()
"""
6 图例

先在 plot 内设置属性 label 给每条线打上标签
然后调用 legend()

legend 默认值为 false 就是不显示
属性
loc 位置 'best' 最好位置, 'lower left' 左下角, 'upper right' 右上角, 等等
"""
plt.legend(loc='best')

"""
7 图层

在 plot 中设置属性 zorder
zorder 决定图像显示优先级
zorder 越大，图层位置越靠向屏幕，否则越低
默认值取决于 axes 类型，无关紧要
"""
plt.plot(X, Y5, label='line5', zorder=3)
# 另外 legend() 调用后 plot 的线不会出现在图例中
plt.legend()

"""
8 多图

fig, ax = plt.subplots(2, 1)
第一个参数表示有几行
第二个参数表示有几列
子图个数为 行数*列数 个
ax 用以保存子图
ax[0] 就是获取第一个子图的权柄
"""
# fig, ax = plt.subplots(2, 1)

# 绘图方法
# 1 使用轴的权柄绘图
# ax[1].plot(X, Y)
# ax[0].plot(X, Y)
# 2 使用 plt.plot
# plt.plot(X+1, Y, color='red', linestyle='-.')

"""
9 输出
'.Big Title.png' 保存文件的名称
dpi 分辨率
"""
plt.savefig('.Big Title.png', dpi=400)

plt.show()
```

