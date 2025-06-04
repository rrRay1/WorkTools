import h5py
import numpy as np
import matplotlib.pyplot as plt
import re

# 全局变量，用于存储找到的波峰
force_peaks_dict = {}


def find_peaks(file_name, gain, offset, threshold1, threshold2):
    """ 查找 .h5 文件中的 Force 数据波峰值。 """
    with h5py.File(file_name, 'r') as file:
        force_data = file['DAQ']['TufADC']['Data']['Force'][:]

    force_data = force_data * gain - offset
    force_peaks = []

    above_threshold = False
    max_value = -np.inf
    for value in force_data:
        if value > threshold1:
            above_threshold = True
            if value > max_value:
                max_value = value
        elif above_threshold and value <= threshold1:
            if max_value > threshold2:
                force_peaks.append(max_value)
            above_threshold = False
            max_value = -np.inf

    if above_threshold and max_value > threshold2:
        force_peaks.append(max_value)

    force_peaks = np.array(force_peaks)
    print(f"Force Peaks for {file_name}: {[int(peak) for peak in force_peaks]}")
    # print(f"Force Peaks for {file_name}: {force_peaks}")
    return force_peaks


def extract_labels(file_name):
    """ 从文件名中提取标签。 """
    match1 = re.search(r"[-+]\d{1,2}C|RT", file_name)
    label1 = match1.group() if match1 else None
    # print(match1)
    match2 = re.search(r"(\d{1,2},\d)V", file_name)
    label2 = match2.group() if match2 else None

    # 确定颜色
    if label1 == "RT":
        color = "green"
    elif label1.startswith('-'):
        # color = "green"
        color = "skyblue"
    else:  # label1必须是以“+”开头
        # color = "green"
        color = "coral"

    return label1, label2, color


def process_files(file_names, gain, offset, threshold1, threshold2):
    """ 处理多个 .h5 文件，提取数据并打上标签。 """
    global force_peaks_dict

    for file_name in file_names:
        force_peaks = find_peaks(file_name, gain, offset, threshold1, threshold2)
        label1, label2, color = extract_labels(file_name)
        force_peaks_dict[file_name] = {'peaks': force_peaks, 'label1': label1, 'label2': label2, 'color': color}


def plot_bar_charts():
    """ 绘制柱状图。 """
    global force_peaks_dict
    plt.figure(figsize=(12, 8))

    for i, (file_name, data) in enumerate(force_peaks_dict.items()):
        peaks = data['peaks']
        if len(peaks) > 0:
            avg_peak = np.mean(peaks)
            max_peak = np.max(peaks)
            min_peak = np.min(peaks)

            # 使用 label1 - label2 作为 x 标签
            x_label = f"{data['label1']} - {data['label2']}"
            # bar = plt.bar(x_label, avg_peak, label=x_label)

            bar = plt.bar(x_label, avg_peak, color=data['color'], label=x_label)  # 使用提取的颜色

            # 在柱子顶部显示平均值
            plt.text(bar[0].get_x() + bar[0].get_width() / 2, avg_peak, f'{int(avg_peak)}',
                     ha='center', va='bottom')

            # 画点表示最大值和最小值
            plt.scatter([bar[0].get_x() + bar[0].get_width() / 2] * 2, [max_peak, min_peak],
                        color='black', zorder=5, label='Max/Min' if i == 0 else "")

        # color.append(colors[i])

    plt.xlabel('Measurements')
    plt.ylabel('Force (N)')
    plt.title('Average Force Peaks from Multiple .h5 Files')
    # plt.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# 使用示例
file_names = [
    # 替换成实际文件路径
    '042_GEE-P145_RT_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_13,5V.h5',
    '042_GEE-P145_RT_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_9,5V.h5',
    '042_GEE-P145_RT_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_15,5V.h5',
    '042_GEE-P145_-30C_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_13,5V.h5',
    '042_GEE-P145_-30C_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_9,5V.h5',
    '042_GEE-P145_-30C_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_15,5V.h5',
    '042_GEE-P145_+80C_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_13,5V.h5',
    '042_GEE-P145_+80C_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_9,5V.h5',
    '042_GEE-P145_+80C_FL-1_LA_FLD_F-R-10Nmm-50mm-s20mm_15,5V.h5',
]

gain = 0.042  # 替换成实际的增益值
offset = 0
threshold1 = 100  # 初始波峰的阈值
threshold2 = 200  # 二次筛选波峰的阈值

# 处理文件并提取数据
process_files(file_names, gain, offset, threshold1, threshold2)

# 绘制柱状图
plot_bar_charts()