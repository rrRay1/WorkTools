#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Author: Rui Ren
# Copyright (c) 2025 CONTINENTAL AUTOMOTIVE. All rights reserved.
# Description: Data filtering and processing script for Gaussian and sliding window filters on HDF5 files.
# Version: 1.0
# Date: 2025-05-31

"""
高斯滤波数据处理工具

用于处理H5文件中的温度数据，应用高斯滤波器，并生成新的H5文件保存处理后的数据。
同时提供原始数据和滤波后数据的可视化比较。
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_h5_dataset(file_path, dataset_path):
    """
    从H5文件中读取指定路径的数据集
    
    参数:
        file_path (str): H5文件路径
        dataset_path (str): 数据集在H5文件中的路径
        
    返回:
        tuple: (数据数组, 属性字典)
    """
    try:
        with h5py.File(file_path, 'r') as file:
            if dataset_path not in file:
                raise KeyError(f"Dataset path '{dataset_path}' does not exist in the file")
            
            dataset = file[dataset_path]
            data = dataset[:]
            # 保存属性
            attrs = dict(dataset.attrs)
            
            return data, attrs
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        raise


def extract_dataset_attributes(attrs, args):
    """
    从数据集属性中提取关键参数，如果属性不存在则使用命令行参数的值
    
    参数:
        attrs (dict): 数据集属性字典
        args (argparse.Namespace): 命令行参数
        
    返回:
        dict: 包含提取的参数的字典
    """
    params = {}
    
    # Extract scaling factor, prefer dataset attribute
    params['scaling_factor'] = attrs.get('scalingFactor', args.scaling_factor)
    logger.info(f"Using scaling factor: {params['scaling_factor']} (from {'dataset attributes' if 'scalingFactor' in attrs else 'command line arguments'})")
    # Extract scaling offset, prefer dataset attribute
    params['scaling_offset'] = attrs.get('scalingOffset', args.scaling_offset)
    logger.info(f"Using scaling offset: {params['scaling_offset']} (from {'dataset attributes' if 'scalingOffset' in attrs else 'command line arguments'})")
    # Extract unit, prefer dataset attribute
    params['unit'] = attrs.get('unit', args.unit)
    logger.info(f"Using unit: {params['unit']} (from {'dataset attributes' if 'unit' in attrs else 'command line arguments'})")
    # Extract alias for title, prefer dataset attribute
    default_alias = os.path.basename(args.input_file).split('.')[0]
    params['alias'] = attrs.get('alias', default_alias)
    logger.info(f"Using alias: {params['alias']} (from {'dataset attributes' if 'alias' in attrs else 'filename'})")
    # Use color settings from command line arguments
    params['original_color'] = args.original_color
    params['filter_color'] = args.filter_color
    logger.info(f"Using original color: {params['original_color']} (from command line arguments)")
    
    return params


def apply_gaussian_filter(data, sigma):
    """
    应用高斯滤波器到数据
    
    参数:
        data (numpy.ndarray): 输入数据数组
        sigma (float): 高斯滤波器的标准差
        
    返回:
        numpy.ndarray: 滤波后的数据
    """
    try:
        return gaussian_filter1d(data, sigma=sigma)
    except Exception as e:
        logger.error(f"Error applying Gaussian filter: {str(e)}")
        raise


def scale_data(data, scaling_factor, scaling_offset):
    """
    根据缩放因子和偏移量转换数据
    
    参数:
        data (numpy.ndarray): 输入数据
        scaling_factor (float): 缩放因子
        scaling_offset (float): 偏移量
        
    返回:
        numpy.ndarray: 缩放后的数据
    """
    return (data / scaling_factor) - scaling_offset


def convert_color_value(color_value):
    """
    转换颜色值为matplotlib可识别的格式
    
    参数:
        color_value: 颜色值，可以是整数(例如42495)或字符串(例如'grey')
        
    返回:
        str或tuple: matplotlib可识别的颜色格式
    """
    # 如果是字符串，直接返回
    if isinstance(color_value, str):
        return color_value
    
    # 如果是整数，转换为RGB元组
    if isinstance(color_value, (int, np.integer)):
        # 将整数拆分为RGB值 (假设格式为0xRRGGBB)
        b = color_value & 0xFF
        g = (color_value >> 8) & 0xFF
        r = (color_value >> 16) & 0xFF
        
        # 转换为0-1范围的RGB元组
        return (r/255.0, g/255.0, b/255.0)
    
    # 如果是其他类型，记录警告并返回默认颜色
    logger.warning(f"Unrecognized color value type: {type(color_value)}, using default color 'grey'")
    return 'grey'


def plot_data(time, data, title, xlabel, ylabel, color, show=True, save_path=None):
    """
    绘制数据图表
    
    参数:
        time (numpy.ndarray): 时间数组
        data (numpy.ndarray): 数据数组
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        color (str): 线条颜色
        show (bool): 是否显示图表
        save_path (str, optional): 保存图表的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(time, original_data, filtered_data, title, xlabel, ylabel, original_color, filter_color, show=True, save_path=None):
    """
    绘制原始数据和滤波后数据的对比图
    
    参数:
        time (numpy.ndarray): 时间数组
        original_data (numpy.ndarray): 原始数据
        filtered_data (numpy.ndarray): 滤波后数据
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        original_color (str): 原始数据线条颜色
        filter_color (str): 滤波后数据线条颜色
        show (bool): 是否显示图表
        save_path (str, optional): 保存图表的路径
    """
    plt.figure(figsize=(12, 7))
    plt.plot(time, original_data, color=original_color, label='Original Data')
    plt.plot(time, filtered_data, color=filter_color, label='Filtered Data')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison chart saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_filtered_h5(original_file, output_file, dataset_path, filtered_data, original_attrs):
    """
    创建新的H5文件保存滤波后的数据
    
    参数:
        original_file (str): 原始H5文件路径
        output_file (str): 输出H5文件路径
        dataset_path (str): 数据集路径
        filtered_data (numpy.ndarray): 滤波后的数据
        original_attrs (dict): 原始数据集的属性
    """
    try:
        # 如果输出文件已存在，先删除
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info(f"Deleted existing output file: {output_file}")
        
        # 创建新文件
        with h5py.File(output_file, 'w') as new_file:
            # 复制原始文件中的所有内容
            with h5py.File(original_file, 'r') as file:
                for key in file.keys():
                    file.copy(key, new_file)
                
                # 复制顶层属性
                for attr_key in file.attrs.keys():
                    new_file.attrs[attr_key] = file.attrs[attr_key]
            
            # 检查数据集是否存在，如果存在则删除
            dataset_parts = dataset_path.split('/')
            current_group = new_file
            
            # 导航到数据集的父组
            for i in range(len(dataset_parts) - 1):
                current_group = current_group[dataset_parts[i]]
            
            # 如果数据集存在，删除它
            if dataset_parts[-1] in current_group:
                del current_group[dataset_parts[-1]]
            
            # 创建新的数据集
            new_dataset = current_group.create_dataset(dataset_parts[-1], data=filtered_data)
            
            # 设置属性
            for key, value in original_attrs.items():
                new_dataset.attrs[key] = value
            
            logger.info(f"Successfully created filtered H5 file: {output_file}")
    except Exception as e:
        logger.error(f"Error saving filtered H5 file: {str(e)}")
        raise


def process_h5_file(args):
    """
    处理H5文件中的数据，应用高斯滤波器，绘制并保存结果
    
    参数:
        args (argparse.Namespace): 命令行参数
    """
    try:
        file_name = args.input_file
        dataset_path = args.dataset_path
        output_file = args.output_file
        gaussian_sigma = args.gaussian_sigma
        show_plots = not args.no_show_plots
        save_plots = args.save_plots
        plots_dir = args.plots_dir
        
        logger.info(f"Processing file: {file_name}")
        
        # 设置输出文件名
        if output_file is None:
            output_file = file_name.replace('.h5', '_filter.h5')
        
        # 设置图表保存目录
        if save_plots and plots_dir is None:
            plots_dir = os.path.dirname(file_name)
        
        # 读取数据
        data, attrs = read_h5_dataset(file_name, dataset_path)
        logger.info(f"Successfully read dataset, shape: {data.shape}")
        
        # 从数据集属性提取参数
        params = extract_dataset_attributes(attrs, args)
        
        # 应用高斯滤波
        filtered_data = apply_gaussian_filter(data, gaussian_sigma)
        logger.info(f"Applied Gaussian filter, sigma={gaussian_sigma}")
        
        # 转换为显示数据
        original_display = scale_data(data, params['scaling_factor'], params['scaling_offset'])
        filtered_display = scale_data(filtered_data, params['scaling_factor'], params['scaling_offset'])
        
        # 创建时间轴
        time = np.arange(0, len(data)) * args.sample_rate
        
        # 准备图表的Y轴标签
        ylabel = f"Temp ({params['unit']})" if params['unit'] else 'Temp'
        
        # 准备标题
        original_title = f"{params['alias']}-Original"
        filter_title = f"{params['alias']}-Filtered"
        comparison_title = f"{params['alias']}-Comparison"
        
        # 绘制图表
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            original_plot_path = os.path.join(plots_dir, f"{os.path.basename(file_name).split('.')[0]}_original.png")
            filtered_plot_path = os.path.join(plots_dir, f"{os.path.basename(file_name).split('.')[0]}_filtered.png")
            comparison_plot_path = os.path.join(plots_dir, f"{os.path.basename(file_name).split('.')[0]}_comparison.png")
        else:
            original_plot_path = filtered_plot_path = comparison_plot_path = None
        
        # 绘制原始数据图
        plot_data(
            time, original_display, original_title, 
            'Time (ms)', ylabel, params['original_color'], 
            show=show_plots, save_path=original_plot_path
        )
        
        # 绘制滤波后的数据图
        plot_data(
            time, filtered_display, filter_title, 
            'Time (ms)', ylabel, params['filter_color'], 
            show=show_plots, save_path=filtered_plot_path
        )
        
        # 绘制对比图
        plot_comparison(
            time, original_display, filtered_display, comparison_title, 
            'Time (ms)', ylabel, params['original_color'], params['filter_color'],
            show=show_plots, save_path=comparison_plot_path
        )
        
        # 保存滤波后的数据
        save_filtered_h5(file_name, output_file, dataset_path, filtered_data, attrs)
        
        logger.info(f"Processing completed: {file_name}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def parse_arguments():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='H5 file data Gaussian filter processing tool')
    
    parser.add_argument('input_file', help='Input H5 file path')
    parser.add_argument('--output-file', '-o', help='Output H5 file path')
    parser.add_argument('--dataset-path', '-d', default='DAQ/TufADC/Data/CoilTemp', help='Dataset path')
    parser.add_argument('--scaling-factor', '-sf', type=float, default=1.0, help='Data scaling factor (fallback, prefer dataset attribute scalingFactor)')
    parser.add_argument('--scaling-offset', '-so', type=float, default=0.0, help='Data offset (fallback, prefer dataset attribute scalingOffset)')
    parser.add_argument('--unit', '-u', default='', help='Data unit (fallback, prefer dataset attribute unit)')
    parser.add_argument('--sample-rate', '-sr', type=float, default=0.001, help='Sample rate (ms)')
    parser.add_argument('--gaussian-sigma', '-gs', type=float, default=5, help='Standard deviation of Gaussian filter')
    parser.add_argument('--original-color', default='grey', help='Original data line color (fallback, prefer dataset attribute color)')
    parser.add_argument('--filter-color', default='green', help='Filtered data line color')
    parser.add_argument('--no-show-plots', action='store_true', help='Do not show plots')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    parser.add_argument('--plots-dir', help='Directory to save plots')
    
    return parser.parse_args()


if __name__ == "__main__":
    # 尝试使用命令行参数，如果没有则使用默认值
    try:
        args = parse_arguments()
        process_h5_file(args)
    except SystemExit:
        # 如果没有提供命令行参数，使用内置的示例值
        logger.info("Running with default parameters...")
        
        # 创建一个模拟的参数对象
        class DefaultArgs:
            def __init__(self):
                self.input_file = '008_SS21_SR_8A_13V_max126C.h5'
                self.output_file = None
                self.dataset_path = 'DAQ/TufADC/Data/ADC1'
                self.scaling_factor = 22.9
                self.scaling_offset = 246
                self.unit = 'degreeCelsius'
                self.sample_rate = 0.00125  # 1.25 ms
                self.gaussian_sigma = 1000
                self.original_color = 'grey'
                self.filter_color = 'green'
                self.no_show_plots = False
                self.save_plots = False
                self.plots_dir = None
        
        default_args = DefaultArgs()
        process_h5_file(default_args)
