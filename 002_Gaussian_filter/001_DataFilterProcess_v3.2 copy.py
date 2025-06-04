#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
                raise KeyError(f"数据集路径 '{dataset_path}' 在文件中不存在")
            
            dataset = file[dataset_path]
            data = dataset[:]
            # 保存属性
            attrs = dict(dataset.attrs)
            
            return data, attrs
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        raise


def extract_dataset_parameters(attrs, file_path, dataset_path):
    """
    从数据集属性中提取处理参数
    
    参数:
        attrs (dict): 数据集属性字典
        file_path (str): H5文件路径（用于日志记录）
        dataset_path (str): 数据集路径（用于日志记录）
        
    返回:
        tuple: (scaling_factor, scaling_offset, unit, sample_rate)
    """
    # 默认值
    scaling_factor = 1.0
    scaling_offset = 0.0
    unit = ''
    sample_rate = 0.001  # 默认1ms
    
    # 尝试从数据集属性中读取
    try:
        # 尝试读取缩放因子
        if 'scalingFactor' in attrs:
            scaling_factor = float(attrs['scalingFactor'])
            logger.info(f"Found scaling factor in dataset: {scaling_factor}")
        elif 'scale' in attrs:
            scaling_factor = float(attrs['scale'])
            logger.info(f"Found scaling factor (scale) in dataset: {scaling_factor}")
        else:
            logger.warning(f"No scaling factor found in dataset {dataset_path}, using default: {scaling_factor}")
        
        # 尝试读取偏移量
        if 'scalingOffset' in attrs:
            scaling_offset = float(attrs['scalingOffset'])
            logger.info(f"Found scaling offset in dataset: {scaling_offset}")
        elif 'offset' in attrs:
            scaling_offset = float(attrs['offset'])
            logger.info(f"Found scaling offset (offset) in dataset: {scaling_offset}")
        else:
            logger.warning(f"No scaling offset found in dataset {dataset_path}, using default: {scaling_offset}")
        
        # 尝试读取单位
        if 'unit' in attrs:
            unit = attrs['unit']
            logger.info(f"Found unit in dataset: {unit}")
        elif 'Unit' in attrs:
            unit = attrs['Unit']
            logger.info(f"Found unit (Unit) in dataset: {unit}")
        else:
            logger.warning(f"No unit found in dataset {dataset_path}, using default: {unit}")
        
        # 尝试读取采样率
        if 'samplingRate' in attrs:
            sample_rate = float(attrs['samplingRate'])
            logger.info(f"Found sampling rate in dataset: {sample_rate}")
        elif 'sampling_rate' in attrs:
            sample_rate = float(attrs['sampling_rate'])
            logger.info(f"Found sampling rate (sampling_rate) in dataset: {sample_rate}")
        elif 'sampleRate' in attrs:
            sample_rate = float(attrs['sampleRate'])
            logger.info(f"Found sampling rate (sampleRate) in dataset: {sample_rate}")
        else:
            # 尝试从文件中获取时间戳数据集来计算采样率
            try:
                with h5py.File(file_path, 'r') as file:
                    # 常见的时间戳数据集路径
                    timestamp_paths = [
                        '/Time',
                        '/Timestamps',
                        '/TimeStamps',
                        '/time',
                        '/timestamps',
                        dataset_path.rsplit('/', 1)[0] + '/Time',
                        dataset_path.rsplit('/', 1)[0] + '/Timestamps'
                    ]
                    
                    for ts_path in timestamp_paths:
                        if ts_path in file:
                            timestamps = file[ts_path][:]
                            if len(timestamps) > 1:
                                # 计算平均采样率
                                avg_diff = np.mean(np.diff(timestamps))
                                sample_rate = avg_diff / 1000.0  # 假设时间戳单位为μs，转换为ms
                                logger.info(f"Calculated sampling rate from timestamps: {sample_rate}ms")
                                break
            except Exception as e:
                logger.warning(f"Could not determine sampling rate from timestamps: {str(e)}")
            
            logger.warning(f"No sampling rate found, using default: {sample_rate}ms")
    
    except Exception as e:
        logger.warning(f"Error extracting parameters from dataset attributes: {str(e)}")
        logger.warning(f"Using default parameters: scaling_factor={scaling_factor}, scaling_offset={scaling_offset}, unit='{unit}', sample_rate={sample_rate}")
    
    return scaling_factor, scaling_offset, unit, sample_rate


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


def plot_comparison(time, original_data, filtered_data, title, xlabel, ylabel, show=True, save_path=None):
    """
    绘制原始数据和滤波后数据的对比图
    
    参数:
        time (numpy.ndarray): 时间数组
        original_data (numpy.ndarray): 原始数据
        filtered_data (numpy.ndarray): 滤波后数据
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        show (bool): 是否显示图表
        save_path (str, optional): 保存图表的路径
    """
    plt.figure(figsize=(12, 7))
    plt.plot(time, original_data, color='grey', label='Original Data')
    plt.plot(time, filtered_data, color='green', label='Filtered Data')
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


def process_h5_file(
        file_name, 
        dataset_path='DAQ/TufADC/Data/CoilTemp',
        output_file=None,
        scaling_factor=None, 
        scaling_offset=None, 
        unit=None, 
        sample_rate=None,
        gaussian_sigma=5,
        original_color='grey',
        filter_color='green',
        original_title='Original Data',
        filter_title='Filtered Data',
        comparison_title='Comparison',
        show_plots=True,
        save_plots=False,
        plots_dir=None,
        auto_params=True):
    """
    处理H5文件中的数据，应用高斯滤波器，绘制并保存结果
    
    参数:
        file_name (str): 输入H5文件路径
        dataset_path (str): 数据集在H5文件中的路径
        output_file (str, optional): 输出H5文件路径，默认为原文件名+_filter.h5
        scaling_factor (float, optional): 数据缩放因子，如果为None且auto_params=True则从数据集属性中读取
        scaling_offset (float, optional): 数据偏移量，如果为None且auto_params=True则从数据集属性中读取
        unit (str, optional): 数据单位，如果为None且auto_params=True则从数据集属性中读取
        sample_rate (float, optional): 采样率（毫秒），如果为None且auto_params=True则从数据集属性中读取
        gaussian_sigma (float): 高斯滤波器的标准差
        original_color (str): 原始数据线条颜色
        filter_color (str): 滤波后数据线条颜色
        original_title (str): 原始数据图表标题
        filter_title (str): 滤波后数据图表标题
        comparison_title (str): 对比图标题
        show_plots (bool): 是否显示图表
        save_plots (bool): 是否保存图表
        plots_dir (str, optional): 保存图表的目录
        auto_params (bool): 是否自动从数据集属性中读取参数
    """
    try:
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
        
        # 如果启用自动参数检测，从数据集属性中提取参数
        if auto_params:
            logger.info("Auto parameter detection enabled, reading parameters from dataset attributes")
            auto_scaling_factor, auto_scaling_offset, auto_unit, auto_sample_rate = extract_dataset_parameters(attrs, file_name, dataset_path)
            
            # 仅在未手动指定参数时使用自动检测的参数
            if scaling_factor is None:
                scaling_factor = auto_scaling_factor
                logger.info(f"Using auto-detected scaling factor: {scaling_factor}")
            else:
                logger.info(f"Using manually specified scaling factor: {scaling_factor}")
                
            if scaling_offset is None:
                scaling_offset = auto_scaling_offset
                logger.info(f"Using auto-detected scaling offset: {scaling_offset}")
            else:
                logger.info(f"Using manually specified scaling offset: {scaling_offset}")
                
            if unit is None:
                unit = auto_unit
                logger.info(f"Using auto-detected unit: '{unit}'")
            else:
                logger.info(f"Using manually specified unit: '{unit}'")
                
            if sample_rate is None:
                sample_rate = auto_sample_rate
                logger.info(f"Using auto-detected sample rate: {sample_rate}ms")
            else:
                logger.info(f"Using manually specified sample rate: {sample_rate}ms")
        else:
            # 使用默认值或手动指定的值
            scaling_factor = 1.0 if scaling_factor is None else scaling_factor
            scaling_offset = 0.0 if scaling_offset is None else scaling_offset
            unit = '' if unit is None else unit
            sample_rate = 0.001 if sample_rate is None else sample_rate
            logger.info(f"Using parameters: scaling_factor={scaling_factor}, scaling_offset={scaling_offset}, unit='{unit}', sample_rate={sample_rate}ms")
        
        # 应用高斯滤波
        filtered_data = apply_gaussian_filter(data, gaussian_sigma)
        logger.info(f"Applied Gaussian filter, sigma={gaussian_sigma}")
        
        # 转换为显示数据
        original_display = scale_data(data, scaling_factor, scaling_offset)
        filtered_display = scale_data(filtered_data, scaling_factor, scaling_offset)
        
        # 创建时间轴
        time = np.arange(0, len(data)) * sample_rate
        
        # 准备图表的Y轴标签
        ylabel = f'Temp ({unit})' if unit else 'Temp'
        
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
            'Time (ms)', ylabel, original_color, 
            show=show_plots, save_path=original_plot_path
        )
        
        # 绘制滤波后的数据图
        plot_data(
            time, filtered_display, filter_title, 
            'Time (ms)', ylabel, filter_color, 
            show=show_plots, save_path=filtered_plot_path
        )
        
        # 绘制对比图
        plot_comparison(
            time, original_display, filtered_display, comparison_title, 
            'Time (ms)', ylabel, show=show_plots, save_path=comparison_plot_path
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
    parser = argparse.ArgumentParser(description='H5文件数据高斯滤波处理工具')
    
    parser.add_argument('input_file', help='输入H5文件路径')
    parser.add_argument('--output-file', '-o', help='输出H5文件路径')
    parser.add_argument('--dataset-path', '-d', default='DAQ/TufADC/Data/CoilTemp', help='数据集路径')
    parser.add_argument('--scaling-factor', '-sf', type=float, help='数据缩放因子（如未指定，从数据集属性中读取）')
    parser.add_argument('--scaling-offset', '-so', type=float, help='数据偏移量（如未指定，从数据集属性中读取）')
    parser.add_argument('--unit', '-u', help='数据单位（如未指定，从数据集属性中读取）')
    parser.add_argument('--sample-rate', '-sr', type=float, help='采样率（毫秒）（如未指定，从数据集属性中读取）')
    parser.add_argument('--gaussian-sigma', '-gs', type=float, default=5, help='高斯滤波器的标准差')
    parser.add_argument('--no-show-plots', action='store_true', help='不显示图表')
    parser.add_argument('--save-plots', action='store_true', help='保存图表')
    parser.add_argument('--plots-dir', help='保存图表的目录')
    parser.add_argument('--no-auto-params', action='store_true', help='禁用自动参数检测')
    
    return parser.parse_args()


if __name__ == "__main__":
    # 尝试使用命令行参数，如果没有则使用默认值
    try:
        args = parse_arguments()
        process_h5_file(
            file_name=args.input_file,
            dataset_path=args.dataset_path,
            output_file=args.output_file,
            scaling_factor=args.scaling_factor,
            scaling_offset=args.scaling_offset,
            unit=args.unit,
            sample_rate=args.sample_rate,
            gaussian_sigma=args.gaussian_sigma,
            show_plots=not args.no_show_plots,
            save_plots=args.save_plots,
            plots_dir=args.plots_dir,
            auto_params=not args.no_auto_params
        )
    except SystemExit:
        # 如果没有提供命令行参数，使用内置的示例值
        logger.info("Running with default parameters...")
        process_h5_file(
            file_name='001_max100dC.h5',
            gaussian_sigma=1000,
            show_plots=True,
            # 不指定scaling_factor, scaling_offset, unit和sample_rate，
            # 让函数自动从数据集中获取这些参数
            auto_params=True
        )
