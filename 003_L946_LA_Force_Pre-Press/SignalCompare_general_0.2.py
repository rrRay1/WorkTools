# -*- coding: utf-8 -*-

# Author: Rui Ren
# Copyright (c) 2025 CONTINENTAL AUTOMOTIVE. All rights reserved.
# Description: HDF5 signal data comparison and visualization tool. Extracts signal data from HDF5 files and visualizes comparisons based on configuration files.
# Version: 1.0
# Date: 2025-05-31

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import rcParams
import os
import glob
import logging
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置matplotlib字体
rcParams['font.family'] = 'sans-serif'

def load_config(config_file):
    """
    加载Excel配置文件
    
    参数:
        config_file (str): 配置文件路径
        
    返回:
        dict: 配置参数字典
    """
    try:
        df_config = pd.read_excel(config_file, sheet_name='config', header=None)
        logger.info(f"成功加载配置信息")
        return {row[0]: row[1] for _, row in df_config.iterrows()}
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise

def get_h5_units(h5_obj, path):
    """
    获取HDF5数据集的单位属性
    
    参数:
        h5_obj: HDF5文件对象
        path (str): 数据集路径
        
    返回:
        str: 单位字符串，如果不存在则返回空字符串
    """
    try:
        return h5_obj[path].attrs.get('unit', '')
    except KeyError:
        return ''

def get_save_filename(base_filename='output_figure', extension='.png'):
    """
    生成自增序号的保存文件名
    
    参数:
        base_filename (str): 基本文件名
        extension (str): 文件扩展名
        
    返回:
        str: 完整的保存文件路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, base_filename)
    
    existing_files = glob.glob(f"{base_path}*{extension}")
    
    if existing_files:
        numbers = []
        for file in existing_files:
            filename = os.path.basename(file)
            if filename == f"{base_filename}{extension}":
                numbers.append(0)
            else:
                try:
                    num_part = filename.replace(base_filename, '').replace(extension, '').strip('_')
                    if num_part:
                        numbers.append(int(num_part))
                except ValueError:
                    continue
        
        next_number = max(numbers) + 1 if numbers else 1
        save_filename = os.path.join(current_dir, f"{base_filename}_{next_number}{extension}")
    else:
        save_filename = os.path.join(current_dir, f"{base_filename}{extension}")
    
    return save_filename

def process_x_axis_data(h5_file, position_path, timestamp_path, position_mode, absolute_mode, start_idx, end_idx):
    """
    处理X轴数据
    
    参数:
        h5_file: HDF5文件对象
        position_path (str): 位置数据路径
        timestamp_path (str): 时间戳数据路径
        position_mode (bool): 是否为位置模式
        absolute_mode (bool): 是否为绝对值模式
        start_idx (int): 起始索引
        end_idx (int): 结束索引
        
    返回:
        tuple: (x_values, x_unit) - X轴值和单位
    """
    if position_mode:
        x_dataset = h5_file[position_path]
        x_abs_unit = get_h5_units(h5_file, position_path)
    else:
        x_dataset = h5_file[timestamp_path]
        x_abs_unit = 's'  # 强制显示秒为单位
    
    # 处理横轴数据缩放
    x_scale = x_dataset.attrs.get('scalingFactor', 1.0)
    x_offset = x_dataset.attrs.get('scalingOffset', 0.0)
    
    # 获取当前区间的横轴数据
    segment_x = (x_dataset[start_idx:end_idx] / x_scale) - x_offset
    
    if position_mode:
        if absolute_mode:
            x_unit = x_abs_unit
            x_values = segment_x
        else:
            x_unit = x_abs_unit
            x_values = segment_x - segment_x[0]  # 位置相对值
            if any(n < 0 for n in x_values):
                x_values = -x_values
    else:
        x_values = (segment_x - segment_x[0]) / 1e6  # 相对时间零点，微秒转秒
        x_unit = 's'
    
    return x_values, x_unit

def process_y_axis_data(h5_file, data_path, start_idx, end_idx):
    """
    处理Y轴数据
    
    参数:
        h5_file: HDF5文件对象
        data_path (str): 数据路径
        start_idx (int): 起始索引
        end_idx (int): 结束索引
        
    返回:
        numpy.ndarray: 处理后的Y轴数据
    """
    data_set = h5_file[data_path]
    
    # 处理数据缩放
    data_scale = data_set.attrs.get('scalingFactor', 1.0)
    data_offset = data_set.attrs.get('scalingOffset', 0.0)
    
    # 应用缩放因子
    return (data_set[start_idx:end_idx] / data_scale) - data_offset

def process_h5_data(config_file):
    """
    主处理函数，根据配置文件处理H5数据并生成可视化图表
    
    参数:
        config_file (str): 配置文件路径
    """
    try:
        # 加载配置
        config = load_config(config_file)
        position_mode = config.get('position_base_enable', 0) == 1
        absolute_mode = config.get('x-axis_absolute_values_enable', 0) == 1
        
        # 初始化画布
        fig, main_ax = plt.subplots(figsize=(18, 10))
        main_ax.set_title(config.get('plot_title', 'Data Visualization'))
        
        # 坐标轴管理系统
        y_axes = {}  # {ylabel: ax}
        axis_offset = 1.0
        
        # 读取Excel文件，获取所有Sheet名称
        data_sheets = pd.ExcelFile(config_file).sheet_names
        
        for sheet in data_sheets:
            if sheet == 'config':
                continue
                
            logger.info(f"处理 {sheet}...")
            df = pd.read_excel(config_file, sheet_name=sheet, header=0)
            
            # 解析文件头
            h5_file_name = df.at[0, 'input_file_name']
            position_path = df.iloc[0]['position_path']
            timestamp_path = df.iloc[0]['timestamp_path']
            
            try:
                with h5py.File(h5_file_name, 'r') as h5_file:
                    # 获取时间索引数组
                    timestamp_array = h5_file[timestamp_path][:]
                    
                    # 处理每个数据条目
                    for idx, row in df.iterrows():
                        if idx == 0 or pd.isna(row['input_data_path']):
                            continue
                        
                        data_path = row['input_data_path']
                        
                        # 获取时间索引
                        start_time = row['input_start_time']
                        end_time = row['input_end_time']
                        start_idx = np.searchsorted(timestamp_array, start_time, side='left')
                        end_idx = np.searchsorted(timestamp_array, end_time, side='right')
                        
                        # 检查索引范围
                        if start_idx >= end_idx:
                            logger.warning(f"无效的时间范围: {start_time} - {end_time}")
                            continue
                        
                        # 处理X轴数据
                        x_values, x_unit = process_x_axis_data(
                            h5_file, position_path, timestamp_path, 
                            position_mode, absolute_mode, start_idx, end_idx
                        )
                        
                        # 处理Y轴数据
                        try:
                            y_data = process_y_axis_data(h5_file, data_path, start_idx, end_idx)
                        except KeyError as e:
                            logger.warning(f"数据路径错误: {data_path}, {str(e)}")
                            continue
                        
                        # 检查数据是否为空
                        if y_data.size == 0 or x_values.size == 0:
                            logger.warning(f"时间区间 ({start_time}, {end_time}) 内无数据.")
                            continue
                        
                        # 动态创建Y轴
                        ylabel = row['input_ylabel']
                        if ylabel not in y_axes:
                            main_ax.yaxis.set_visible(False)  # 隐藏左侧Y轴刻度
                            main_ax.spines['left'].set_visible(False)  # 隐藏左侧轴线  
                            new_ax = main_ax.twinx()
                            new_ax.spines.right.set_position(("axes", axis_offset))
                            axis_offset += 0.1
                            new_ax.set_ylabel(ylabel, color='black')
                            new_ax.tick_params(axis='y', labelcolor='black')
                            y_axes[ylabel] = new_ax
                        
                        # 获取当前行的颜色和线型
                        color = row['input_color'] if not pd.isna(row['input_color']) else None
                        linestyle = row['input_linestyle'] if not pd.isna(row['input_linestyle']) else '-'
                        
                        # 绘制数据
                        y_axes[ylabel].plot(
                            x_values, 
                            y_data, 
                            color=color,
                            linestyle=linestyle,
                            label=row['input_legend_label'],
                            linewidth=1.5
                        )
                        
            except (KeyError, OSError) as e:
                logger.error(f"处理文件 {h5_file_name} 时出错: {str(e)}")
                continue
        
        # 设置X轴标签
        axis_type = "Position" if position_mode else "Time"
        value_type = "Absolute" if absolute_mode else "Relative"
        main_ax.set_xlabel(f"{value_type} {axis_type} ({x_unit})")
        
        # 合并图例
        handles, labels = [], []
        h_main, l_main = main_ax.get_legend_handles_labels()
        handles.extend(h_main)
        labels.extend(l_main)
        
        for ax in y_axes.values():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        
        if handles and labels:
            main_ax.legend(
                handles, labels,
                loc='upper left',
                bbox_to_anchor=(0, 1.2),
                ncol=3,
                fontsize=8,
                framealpha=0.7
            )
        
        plt.tight_layout()
        
        # 保存图形
        logger.info("开始保存图形...")
        save_filename = get_save_filename()
        
        try:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            logger.info(f"图形已成功保存为: {save_filename}")
        except Exception as e:
            logger.error(f"保存图形时出错: {str(e)}")
        
        # 显示图形
        plt.show()
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import sys
    
    # 使用命令行参数或默认值
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configurations_general.xlsx"
    logger.info(f"使用配置文件: {config_file}")
    
    process_h5_data(config_file)