#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Author: Rui Ren
# Copyright (c) 2025 CONTINENTAL AUTOMOTIVE. All rights reserved.
# Description: Data processing script for force peak finding and analysis in HDF5 files.
# Version: 1.0
# Date: 2025-05-31

"""

此脚本用于从h5文件中提取力值数据，检测峰值，并生成统计图表。
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import argparse
from typing import List, Dict, Tuple, Optional, Any


class ForcePeakAnalyzer:
    """力值数据峰值分析类"""
    
    def __init__(self, threshold1: float = 100, threshold2: float = 200, 
                 default_gain: float = 0.042, default_offset: float = 0,
                 default_unit: str = "N"):
        """
        初始化力值数据分析器
        
        Args:
            threshold1: 初始峰值检测阈值
            threshold2: 二次峰值筛选阈值
            default_gain: 默认增益系数，当无法从数据属性中读取时使用
            default_offset: 默认偏移值，当无法从数据属性中读取时使用
            default_unit: 默认单位，当无法从数据属性中读取时使用
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.default_gain = default_gain
        self.default_offset = default_offset
        self.default_unit = default_unit
        self.force_peaks_dict: Dict[str, Dict[str, Any]] = {}
        self.units: Dict[str, str] = {}  # 存储每个文件的单位
    
    def find_peaks(self, file_path: str) -> np.ndarray:
        """
        查找h5文件中的力数据峰值，自动从数据属性中获取缩放因子和偏移量
        
        Args:
            file_path: h5文件路径
            
        Returns:
            检测到的峰值数组
        """
        try:
            with h5py.File(file_path, 'r') as file:
                force_dataset = file['DAQ']['TufADC']['Data']['Force']
                force_data = force_dataset[:]
                
                # 从数据属性中读取缩放因子、偏移量和单位
                scaling_factor = force_dataset.attrs.get('scalingFactor', self.default_gain)
                scaling_offset = force_dataset.attrs.get('scalingOffset', self.default_offset)
                unit = force_dataset.attrs.get('unit', self.default_unit)
                
                # 存储单位信息，用于后续绘图
                self.units[file_path] = unit
                
                print(f"File {os.path.basename(file_path)} Scaling Factor: {scaling_factor}, Offset: {scaling_offset}, Unit: {unit}")
        except (FileNotFoundError, KeyError) as e:
            print(f"Error: Failed to read data from {file_path}: {e}")
            return np.array([])
            
        # 应用缩放因子和偏移量
        force_data = force_data / scaling_factor - scaling_offset
        force_peaks = []
        
        # 峰值检测算法
        above_threshold = False
        max_value = -np.inf
        for value in force_data:
            if value > self.threshold1:
                above_threshold = True
                if value > max_value:
                    max_value = value
            elif above_threshold and value <= self.threshold1:
                if max_value > self.threshold2:
                    force_peaks.append(max_value)
                above_threshold = False
                max_value = -np.inf
        
        # 检查最后一个潜在峰值
        if above_threshold and max_value > self.threshold2:
            force_peaks.append(max_value)
        
        force_peaks = np.array(force_peaks)
        print(f"File {os.path.basename(file_path)} Force Peak: {[int(peak) for peak in force_peaks]}")
        return force_peaks
    
    def extract_labels(self, file_name: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        从文件名中提取标签信息
        
        Args:
            file_name: 文件名
            
        Returns:
            (温度标签, 电压标签, 颜色)元组
        """
        # 提取温度标签
        temp_match = re.search(r"[-+]\d{1,2}C|RT", file_name)
        temp_label = temp_match.group() if temp_match else None
        
        # 提取电压标签
        voltage_match = re.search(r"(\d{1,2},\d)V", file_name)
        voltage_label = voltage_match.group() if voltage_match else None
        
        # 根据温度确定颜色
        if temp_label == "RT":
            color = "green"
        elif temp_label and temp_label.startswith('-'):
            color = "skyblue"
        elif temp_label and temp_label.startswith('+'):
            color = "coral"
        else:
            color = "gray"  # 默认颜色
            
        return temp_label, voltage_label, color
    
    def process_files(self, file_paths: List[str]) -> None:
        """
        处理多个h5文件
        
        Args:
            file_paths: h5文件路径列表
        """
        self.force_peaks_dict.clear()
        self.units.clear()
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File does not exist - {file_path}")
                continue
                
            file_name = os.path.basename(file_path)
            force_peaks = self.find_peaks(file_path)
            temp_label, voltage_label, color = self.extract_labels(file_name)
            
            self.force_peaks_dict[file_path] = {
                'peaks': force_peaks,
                'label1': temp_label,
                'label2': voltage_label,
                'color': color,
                'unit': self.units.get(file_path, self.default_unit)
            }
    
    def plot_bar_charts(self, save_path: Optional[str] = None) -> None:
        """
        绘制峰值柱状图
        
        Args:
            save_path: 图表保存路径，如果为None则显示图表
        """
        if not self.force_peaks_dict:
            print("Error: No data available for plotting")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 确定使用的单位，默认使用第一个文件的单位
        common_unit = next(iter(self.force_peaks_dict.values()))['unit'] if self.force_peaks_dict else self.default_unit
        
        for i, (file_path, data) in enumerate(self.force_peaks_dict.items()):
            peaks = data['peaks']
            if len(peaks) == 0:
                continue
                
            avg_peak = np.mean(peaks)
            max_peak = np.max(peaks)
            min_peak = np.min(peaks)
            
            # 使用温度-电压作为x标签
            x_label = f"{data['label1']} - {data['label2']}"
            bar = plt.bar(x_label, avg_peak, color=data['color'], label=x_label)
            
            # 显示平均值
            plt.text(bar[0].get_x() + bar[0].get_width() / 2, avg_peak, f'{int(avg_peak)}',
                     ha='center', va='bottom')
            
            # 显示最大值和最小值
            plt.scatter([bar[0].get_x() + bar[0].get_width() / 2] * 2, 
                        [max_peak, min_peak],
                        color='black', zorder=5, 
                        label='Max/Min' if i == 0 else "")
        
        plt.xlabel('Measurement Conditions')
        plt.ylabel(f'Force ({common_unit})')
        plt.title(f'Average Force Peaks of Multiple Conditions')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Chart saved to: {save_path}")
        else:
            plt.show()

    def get_unit_info(self) -> Dict[str, str]:
        """
        获取所有文件的单位信息
        
        Returns:
            文件路径到单位的映射字典
        """
        return self.units


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='力值数据峰值检测与分析工具')
    
    parser.add_argument('-f', '--files', nargs='+', required=False,
                        help='要处理的h5文件路径列表')
    parser.add_argument('-t1', '--threshold1', type=float, default=100,
                        help='初始峰值检测阈值 (默认: 100)')
    parser.add_argument('-t2', '--threshold2', type=float, default=200,
                        help='二次峰值筛选阈值 (默认: 200)')
    parser.add_argument('-g', '--default_gain', type=float, default=0.042,
                        help='默认增益系数，当无法从数据属性中读取时使用 (默认: 0.042)')
    parser.add_argument('-o', '--default_offset', type=float, default=0,
                        help='默认偏移值，当无法从数据属性中读取时使用 (默认: 0)')
    parser.add_argument('-u', '--default_unit', type=str, default='N',
                        help='默认单位，当无法从数据属性中读取时使用 (默认: N)')
    parser.add_argument('-s', '--save', type=str,
                        help='保存图表的文件路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 如果没有命令行参数，使用默认示例文件
    args = parse_arguments()
    
    if not args.files:
        # 使用示例文件
        file_names = [
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
        print("Processing files...")
    else:
        file_names = args.files
        
    # 创建分析器
    analyzer = ForcePeakAnalyzer(
        threshold1=args.threshold1,
        threshold2=args.threshold2,
        default_gain=args.default_gain,
        default_offset=args.default_offset,
        default_unit=args.default_unit
    )
    
    # 处理文件
    analyzer.process_files(file_names)
    
    # 绘制图表
    analyzer.plot_bar_charts(save_path=args.save)


if __name__ == "__main__":
    main()