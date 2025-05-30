# -*- coding: utf-8 -*-
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import rcParams
import os
import glob
import sys

rcParams['font.family'] = 'sans-serif'

def load_config(config_file):
    """增强型配置加载"""
    df_config = pd.read_excel(config_file, sheet_name='config', header=None)
    # debug
    print(f"\nCurrent [{'config'}] info:")
    print(df_config.head())       # 显示前5行
    return {row[0]: row[1] for _, row in df_config.iterrows()}
    


def get_h5_units(h5_obj, path):
    """获取HDF5数据集的单位属性"""
    try:
        return h5_obj[path].attrs.get('unit', '')
    except KeyError:
        return ''

def process_h5_data(config_file):
    config = load_config(config_file)
    position_mode = config.get('position_base_enable', 0) == 1
    absolute_mode = config.get('x-axis_absolute_values_enable', 0) == 1

    # 初始化画布
    fig, main_ax = plt.subplots(figsize=(18, 10))
    main_ax.set_title(config.get('plot_title', 'Data Visualization'))
    
    # 坐标轴管理系统
    y_axes = {}  # {ylabel: ax}
    axis_offset = 1.0

    # xls = pd.ExcelFile(config_file)
    # data_sheets = [s for s in xls.sheet_names if s.startswith('H5File')]

    # 读取Excel文件，获取所有Sheet名称
    data_sheets = pd.ExcelFile(config_file).sheet_names


    for sheet in data_sheets:
        if sheet == 'config':
            continue
        print(f"\nProcessing {sheet}...")
        df = pd.read_excel(config_file, sheet_name=sheet, header=0)
        
        # debug
        print(f"\nCurrent [{sheet}] info:")
        print(df.head())       # 显示前5行


        # 解析文件头
        h5_file_name = df.at[0, 'input_file_name']
        # group_path = df.at[0, 'group_path']
        # group_path = df.at[0, 'group_path']
        
        # base_group = group_path if group_path else ''

        # print(f"\ninput_file_name: {h5_file_name}")
        # print(f"\ngroup_path: {group_path}")
        

        # # 构建完整数据路径
        # timestamp_path = f"{base_group}/{df.at[0, 'timestamp_path']}"
        # position_path = f"{base_group}/{df.at[0, 'position_path']}"

        position_path = df.iloc[0]['position_path']
        timestamp_path = df.iloc[0]['timestamp_path']

        print(f"\ntimestamp_path: {timestamp_path}")
        print(f"\nposition_path: {position_path}")

        # input(">>> Script paused, press Enter to continue...")
        # def print_h5_structure(name):
        #     print(name)


    
        try:
            with h5py.File(h5_file_name, 'r') as h5_file:
                # ========== 横轴数据处理 ==========
                # 获取基准数据集
                
                if position_mode:
                    #print(f"Current path: {position_path}") 
                    x_dataset = h5_file[position_path]
                    # print(f"\nx_dataset: [{x_dataset}]")
                    # input(">>> Script paused, press Enter to continue...")
                    x_abs_unit = get_h5_units(h5_file, position_path)
                    print(f"\nx_abs_unit: {x_abs_unit}")
                else:
                    x_dataset = h5_file[timestamp_path]
                    # x_abs_unit = get_h5_units(h5_file, timestamp_path)  # 时间原始数据默认为微秒
                    x_abs_unit = 's'  # 强制显示秒为单位
                    
                    print(f"\nx_abs_unit: {x_abs_unit}")

                # print(f"\nx_dataset_h5: {h5_file[position_path][10000: 15000]}")           
                print(f"\nx_dataset_shape: {x_dataset.shape}")
                # print(f"\nx_dataset: {x_dataset[10000:15000]}")
                # input(">>> Script paused, press Enter to continue...")
                

                # 处理横轴数据缩放
                x_scale = x_dataset.attrs.get('scalingFactor', 1.0)
                x_offset = x_dataset.attrs.get('scalingOffset', 0.0)


                # input(">>> Script paused, press Enter to continue...")




                # 设置横轴标签
                axis_type = "Position" if position_mode else "Time"
                value_type = "Absolute" if absolute_mode else "Relative"
                # main_ax.set_xlabel(f"{value_type} {axis_type} ({x_unit})")

                # ========== 处理每个数据条目 ==========
                for idx, row in df.iterrows():
                    if idx == 0 or pd.isna(row['input_data_path']):
                        continue

                    data_path = f"{row['input_data_path']}"
                    data_set = h5_file[data_path]
                    
                    # 获取时间索引

                    timestamp_array = h5_file[timestamp_path][:]

                    start_time = row['input_start_time']
                    end_time = row['input_end_time']
                    start_idx = np.searchsorted(timestamp_array, start_time, side='left')
                    end_idx = np.searchsorted(timestamp_array, end_time, side='right')

                    # 处理数据缩放
                    data_scale = data_set.attrs.get('scalingFactor', 1.0)
                    data_offset = data_set.attrs.get('scalingOffset', 0.0)

                    print(f"\ndata_set_name: {data_set.attrs.get('name', 'unknown')}")
                    print(f"data_set_unit: {data_set.attrs.get('unit', 'unknown')}")


                    y_data = (data_set[start_idx:end_idx] / data_scale) - data_offset


                    print(f"y_data_shape: {y_data.shape}")
                    # print(f"y_data: {y_data[0:200]}")

                    # 获取当前区间的横轴数据

                    segment_x = (x_dataset[start_idx:end_idx] / x_scale) - x_offset


                    if position_mode:
                        ######################################################
                        ################ Position_based_mode #################
                        ######################################################
                        print(f"\n######################################################\n################ Position_based_mode #################\n######################################################")
                        if absolute_mode:
                            print(f"################ Absolute_mode #################")
                            x_unit = x_abs_unit
                            x_values = segment_x
                        else:
                            print(f"################ Relative_mode #################")
                            x_unit = x_abs_unit
                            x_values = segment_x - segment_x[0] # 位置相对值
                            if any(n < 0 for n in x_values):
                                x_values = -x_values
                        
                    else:
                        
                        ######################################################
                        ################ Time_based_mode #################
                        ######################################################
                        print(f"\n######################################################\n################ Time_based_mode #################\n######################################################")

                        x_values = (segment_x - segment_x[0]) / 1e6  # 相对时间零点，微秒转秒
                        x_unit = 's'



                    print(f"\nx_dataset_name: {x_dataset.attrs.get('name', 'unknown')}")
                    print(f"\nx_dataset_unit: {x_unit}")
                    print(f"\nsegment_x_shape: {segment_x.shape}")
                    # print(f"\nsegment_x: {segment_x[0:200]}")

                    # 检查数据是否为空
                    if y_data.size == 0 or segment_x.size == 0:
                        print(f"Warning: No data in interval ({start_time}, {end_time}).")
                        continue

                    # ========== 动态创建Y轴 ==========
                    ylabel = row['input_ylabel']
                    if ylabel not in y_axes:
                        main_ax.yaxis.set_visible(False)  # 隐藏左侧Y轴刻度
                        main_ax.spines['left'].set_visible(False)  # 隐藏左侧轴线  
                        new_ax = main_ax.twinx()
                        new_ax.spines.right.set_position(("axes", axis_offset))
                        axis_offset += 0.1
                        new_ax.set_ylabel(ylabel, color='black')
                        new_ax.tick_params(axis='y', labelcolor='black')
                        y_axes[ylabel] = new_ax  # 直接存储Axes对象

                    # 获取当前行的颜色和线型
                    color = (row['input_color'] if not pd.isna(row['input_color']) 
                            else (random.random(), random.random(), random.random()))
                    linestyle = row['input_linestyle'] if not pd.isna(row['input_linestyle']) else '-'

                    # 绘制数据
                    y_axes[ylabel].plot(
                        segment_x, 
                        y_data, 
                        color=color,
                        linestyle=linestyle,
                        label=row['input_legend_label'],
                        linewidth=1.5
                    )
        except KeyError as e:
            print(f"Critical Error: Path {position_path} not found in {h5_file_name}")
            continue


    
    # ========== 合并图例 ==========
    handles, labels = [], []
    # 主轴的图例句柄（可能为空）
    h_main, l_main = main_ax.get_legend_handles_labels()
    handles.extend(h_main)
    labels.extend(l_main)
    
    # 遍历所有动态创建的Y轴
    for ax in y_axes.values():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    main_ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(0, 1.2),
        ncol=3,
        fontsize=8,
        framealpha=0.7
    )

    plt.tight_layout()
    
    print("\n开始保存图形...")
    # 检测现有图片文件并自动增加序号
    base_filename = 'output_figure'
    extension = '.png'
    
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, base_filename)
    
    print(f"当前目录: {current_dir}")
    existing_files = glob.glob(f"{base_path}*{extension}")
    print(f"找到的现有文件: {existing_files}")
    
    if existing_files:
        # 提取现有文件的序号
        numbers = []
        for file in existing_files:
            # 尝试从文件名中提取序号
            filename = os.path.basename(file)
            print(f"处理文件: {filename}")
            if filename == f"{base_filename}{extension}":
                numbers.append(0)  # 没有序号的文件视为0
                print(f"  基本文件名匹配，添加序号 0")
            else:
                try:
                    # 提取类似 output_figure_1.png 中的数字部分
                    num_part = filename.replace(base_filename, '').replace(extension, '').strip('_')
                    if num_part:
                        numbers.append(int(num_part))
                        print(f"  提取序号: {num_part}")
                except ValueError:
                    print(f"  无法提取序号")
                    continue
        
        # 找到最大序号并加1
        next_number = max(numbers) + 1 if numbers else 1
        save_filename = os.path.join(current_dir, f"{base_filename}_{next_number}{extension}")
        print(f"使用序号 {next_number}")
    else:
        save_filename = os.path.join(current_dir, f"{base_filename}{extension}")
        print("没有找到现有文件，使用基本文件名")
    
    try:
        # 保存图形到文件
        print(f"尝试保存图形到: {save_filename}")
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"图形已成功保存为: {save_filename}")
        
        # 验证文件是否已创建
        if os.path.exists(save_filename):
            print(f"文件已成功创建，大小: {os.path.getsize(save_filename)} 字节")
        else:
            print(f"警告: 文件似乎没有被创建!")
    except Exception as e:
        print(f"保存图形时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 显示图形
    print("尝试显示图形...")
    try:
        plt.show()
        print("plt.show() 已执行")
    except Exception as e:
        print(f"显示图形时出错: {str(e)}")

# 使用示例
config_file = "configurations_general.xlsx"
process_h5_data(config_file)