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
    """Force data peak value analysis class"""
    
    def __init__(self, threshold1: float = 100, threshold2: float = 200, 
                 default_gain: float = 0.042, default_offset: float = 0,
                 default_unit: str = "N"):
        """
        Initialize the force data analyzer
        
        Args:
            threshold1: Initial peak detection threshold
            threshold2: Secondary peak filtering threshold
            default_gain: Default gain coefficient, used when not available in data attributes
            default_offset: Default offset, used when not available in data attributes
            default_unit: Default unit, used when not available in data attributes
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.default_gain = default_gain
        self.default_offset = default_offset
        self.default_unit = default_unit
        self.force_peaks_dict: Dict[str, Dict[str, Any]] = {}
        self.units: Dict[str, str] = {}  # Store unit of each file
    
    def find_peaks(self, file_path: str) -> np.ndarray:
        """
        查找h5文件中的力数据峰值，自动从数据属性中获取缩放因子和偏移量
        
        Args:
            file_path: h5 file path
            
        Returns:
            Detected peaks array
        """
        try:
            with h5py.File(file_path, 'r') as file:
                force_dataset = file['DAQ']['TufADC']['Data']['Force']
                force_data = force_dataset[:]
                
                # Read scaling factor, offset, and unit from data attributes
                scaling_factor = force_dataset.attrs.get('scalingFactor', self.default_gain)
                scaling_offset = force_dataset.attrs.get('scalingOffset', self.default_offset)
                unit = force_dataset.attrs.get('unit', self.default_unit)
                
                # Store unit information for later plotting
                self.units[file_path] = unit
                
                print(f"File {os.path.basename(file_path)} Scaling Factor: {scaling_factor}, Offset: {scaling_offset}, Unit: {unit}")
        except (FileNotFoundError, KeyError) as e:
            print(f"Error: Failed to read data from {file_path}: {e}")
            return np.array([])
            
        # Apply scaling factor and offset
        force_data = force_data / scaling_factor - scaling_offset
        force_peaks = []
        
        # Peak detection algorithm
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
        
        # Check the last potential peak
        if above_threshold and max_value > self.threshold2:
            force_peaks.append(max_value)
        
        force_peaks = np.array(force_peaks)
        print(f"File {os.path.basename(file_path)} Force Peak: {[int(peak) for peak in force_peaks]}")
        return force_peaks
    
    def extract_labels(self, file_name: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        从文件名中提取标签信息
        
        Args:
            file_name: File name
            
        Returns:
            (Temperature label, Voltage label, Color) tuple
        """
        # Extract temperature label
        temp_match = re.search(r"[-+]\d{1,2}C|RT", file_name)
        temp_label = temp_match.group() if temp_match else None
        
        # Extract voltage label
        voltage_match = re.search(r"(\d{1,2},\d)V", file_name)
        voltage_label = voltage_match.group() if voltage_match else None
        
        # Determine color based on temperature
        if temp_label == "RT":
            color = "green"
        elif temp_label and temp_label.startswith('-'):
            color = "skyblue"
        elif temp_label and temp_label.startswith('+'):
            color = "coral"
        else:
            color = "gray"  # Default color
            
        return temp_label, voltage_label, color
    
    def process_files(self, file_paths: List[str]) -> None:
        """
        处理多个h5文件
        
        Args:
            file_paths: List of h5 file paths
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
        Plot peak value bar chart
        
        Args:
            save_path: Chart save path, if None, display the chart
        """
        if not self.force_peaks_dict:
            print("Error: No data available for plotting")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Determine the unit to use, default to the first file's unit
        common_unit = next(iter(self.force_peaks_dict.values()))['unit'] if self.force_peaks_dict else self.default_unit
        
        for i, (file_path, data) in enumerate(self.force_peaks_dict.items()):
            peaks = data['peaks']
            if len(peaks) == 0:
                continue
                
            avg_peak = np.mean(peaks)
            max_peak = np.max(peaks)
            min_peak = np.min(peaks)
            
            # Use temperature-voltage as x label
            x_label = f"{data['label1']} - {data['label2']}"
            bar = plt.bar(x_label, avg_peak, color=data['color'], label=x_label)
            
            # Display average value
            plt.text(bar[0].get_x() + bar[0].get_width() / 2, avg_peak, f'{int(avg_peak)}',
                     ha='center', va='bottom')
            
            # Display max and min values
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
            Mapping of file path to unit
        """
        return self.units


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Force data peak detection and analysis tool')
    
    parser.add_argument('-f', '--files', nargs='+', required=False,
                        help='List of h5 file paths to process')
    parser.add_argument('-t1', '--threshold1', type=float, default=100,
                        help='Initial peak detection threshold (default: 100)')
    parser.add_argument('-t2', '--threshold2', type=float, default=200,
                        help='Secondary peak filtering threshold (default: 200)')
    parser.add_argument('-g', '--default_gain', type=float, default=0.042,
                        help='Default gain coefficient, used when not available in data attributes (default: 0.042)')
    parser.add_argument('-o', '--default_offset', type=float, default=0,
                        help='Default offset, used when not available in data attributes (default: 0)')
    parser.add_argument('-u', '--default_unit', type=str, default='N',
                        help='Default unit, used when not available in data attributes (default: N)')
    parser.add_argument('-s', '--save', type=str,
                        help='File path to save the chart')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Use default example files if no command line arguments are provided
    args = parse_arguments()
    
    if not args.files:
        # Use example files
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
        
    # Create analyzer
    analyzer = ForcePeakAnalyzer(
        threshold1=args.threshold1,
        threshold2=args.threshold2,
        default_gain=args.default_gain,
        default_offset=args.default_offset,
        default_unit=args.default_unit
    )
    
    # Process files
    analyzer.process_files(file_names)
    
    # Plot charts
    analyzer.plot_bar_charts(save_path=args.save)


if __name__ == "__main__":
    main()