#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 ref.json 脚本

用于在不运行模型调用的情况下，将指定目录下各 ref.json 的坐标范围、
操作和状态流转关系可视化到汇总图片中。

使用方法:
    python visualize_ref.py --data_dir ./test_data/offline
    python visualize_ref.py --data_dir ./test_data/offline/好评_98  # 批量处理所有子目录

作为模块使用:
    from utils.visualize_ref import TestVisualizer
    visualizer = TestVisualizer()
    visualizer.visualize_task(steps_record, output_dir)
"""

import os
import json
import argparse
import logging
import textwrap
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 使用非交互后端

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置
ACTION_COLORS = {
    'CLICK': 'cyan',
    'TYPE': 'orange',
    'SCROLL': 'green',
    'COMPLETE': 'purple',
    'OPEN': 'yellow',
}

BRANCH_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

# Agent 操作可视化颜色配置
AGENT_COLORS = {
    'correct': '#00FF00',    # 绿色 - 正确操作
    'incorrect': '#FF0000',  # 红色 - 错误操作
    'click_marker': '#FF00FF',  # 紫色 - 点击标记
    'scroll_arrow': '#FFA500',   # 橙色 - 滑动箭头
}

# 模块级别：只获取 logger，不配置（避免导入时的副作用）
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='可视化 ref.json 中的坐标范围和操作',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_ref.py --data_dir ./test_data/offline/好评_98/dazhongdianping_lsh_scene_0
  python visualize_ref.py --data_dir ./test_data/offline/好评_98 --batch
  python visualize_ref.py --data_dir ./test_data/offline/好评_98 --output ./output_vis.png
        """
    )
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        required=True,
        help='包含 ref.json 和 screenshot 文件夹的数据目录'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出图片路径（默认在数据目录下生成 ref_visualization.png）'
    )
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='批量模式：处理 data_dir 下的所有子目录'
    )
    parser.add_argument(
        '--max_cols',
        type=int,
        default=5,
        help='每行最多显示的步骤数（默认：5）'
    )
    parser.add_argument(
        '--fig_width',
        type=int,
        default=30,
        help='图片宽度（默认：30）'
    )
    parser.add_argument(
        '--show_flow',
        action='store_true',
        default=True,
        help='显示状态流转关系（默认启用）'
    )
    return parser.parse_args()


def load_ref_json(ref_path: str) -> Optional[Dict[str, Any]]:
    """加载并解析 ref.json 文件"""
    try:
        with open(ref_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"找不到文件: {ref_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {ref_path}, 错误: {e}")
        return None


def extract_metadata(ref_data: Dict[str, Any]) -> Dict[str, Any]:
    """提取元数据"""
    # 优先使用 0-0 键，其次使用 0 键
    if '0-0' in ref_data:
        meta = ref_data['0-0']
    elif '0' in ref_data and isinstance(ref_data['0'], dict):
        meta = ref_data['0']
    else:
        meta = {}

    return {
        'instruction': meta.get('instruction', '未知指令'),
        'app': meta.get('app', '未知应用'),
        'screen_shape': meta.get('screen_shape', [1080, 1920]),
        'max_steps': meta.get('max_steps', 20),
    }


def extract_states(ref_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """提取所有状态节点"""
    states = {}
    for key, value in ref_data.items():
        # 跳过元数据键
        if key in ['0-0']:
            continue
        # 只处理状态键（数字或数字-数字格式）
        if isinstance(value, list):
            states[key] = value
    return states


def get_state_order(states: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """
    获取状态的有序列表
    按照步骤数字排序，处理分支情况如 6-1, 6-2
    """
    def sort_key(state_key: str) -> Tuple[int, ...]:
        parts = state_key.split('-')
        return tuple(int(p) for p in parts)

    return sorted(states.keys(), key=sort_key)


def find_screenshot(data_dir: str, state_key: str) -> Optional[str]:
    """
    查找对应状态的截图
    优先查找 screenshot 子目录，如果没有则在 data_dir 中查找
    """
    # 尝试不同的命名格式
    possible_names = [
        f"{state_key}.png",
        f"{state_key}.jpg",
        f"{state_key}.jpeg",
        f"step_{state_key}.png",
        f"step_{state_key}.jpg",
    ]

    # 对于纯数字状态键，添加零填充格式
    try:
        state_num = int(state_key)
        possible_names.extend([
            f"{state_num:03d}.png",
            f"{state_num:03d}.jpg",
        ])
    except ValueError:
        pass

    # 首先尝试在 screenshot 子目录中查找
    screenshot_dir = os.path.join(data_dir, 'screenshot')
    if os.path.exists(screenshot_dir):
        for name in possible_names:
            path = os.path.join(screenshot_dir, name)
            if os.path.exists(path):
                return path

        # 尝试列表中的文件，找最接近的
        try:
            files = sorted([f for f in os.listdir(screenshot_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))])

            # 尝试匹配数字部分
            state_parts = state_key.split('-')
            state_num = int(state_parts[0])
            for f in files:
                # 提取文件名中的数字
                import re
                numbers = re.findall(r'\d+', f)
                if numbers and int(numbers[0]) == state_num:
                    return os.path.join(screenshot_dir, f)
        except (ValueError, IndexError):
            pass

    # 然后直接在 data_dir 中查找
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    # 尝试在 data_dir 中模糊匹配
    try:
        files = sorted([f for f in os.listdir(data_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 尝试精确匹配状态键（如 "6-1"）
        for f in files:
            name_without_ext = os.path.splitext(f)[0]
            if name_without_ext == state_key:
                return os.path.join(data_dir, f)

        # 尝试匹配数字部分
        state_parts = state_key.split('-')
        state_num = int(state_parts[0])
        for f in files:
            name_without_ext = os.path.splitext(f)[0]
            # 提取文件名中的数字
            import re
            numbers = re.findall(r'\d+', name_without_ext)
            if numbers and int(numbers[0]) == state_num:
                return os.path.join(data_dir, f)
    except (ValueError, IndexError):
        pass

    return None


def plot_click_region(ax, x_range: List[int], y_range: List[int],
                      color: str = 'cyan', linewidth: int = 6,
                      alpha: float = 0.8, label: Optional[str] = None):
    """在坐标轴上绘制 CLICK 的矩形区域"""
    x1, x2 = x_range
    y1, y2 = y_range

    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        alpha=alpha,
        linestyle='-'
    )
    ax.add_patch(rect)

    # 在矩形中心添加标签
    if label:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        ax.text(
            center_x, center_y,
            label,
            color='white',
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(
                facecolor=color,
                alpha=0.9,
                edgecolor='none',
                boxstyle='round,pad=0.3'
            )
        )

    return ax


def plot_action_on_axis(ax, action: Dict[str, Any],
                        screen_shape: List[int],
                        branch_idx: int = 0,
                        show_details: bool = True):
    """在坐标轴上绘制单个动作"""
    action_type = action.get('action', 'UNKNOWN')
    params = action.get('params', {})
    next_state = action.get('next', '')

    color = BRANCH_COLORS[branch_idx % len(BRANCH_COLORS)]

    if action_type == 'CLICK':
        # 使用 x_real/y_real 作为实际屏幕坐标
        x_range = params.get('x_real', params.get('x', [0, 100]))
        y_range = params.get('y_real', params.get('y', [0, 100]))
        plot_click_region(ax, x_range, y_range, color=color, label=f'C{branch_idx+1}')

    elif action_type == 'SCROLL':
        # 绘制滚动箭头（在屏幕中央）
        screen_w, screen_h = screen_shape
        center_x = screen_w / 2
        top_y = screen_h * 0.3
        bottom_y = screen_h * 0.7

        ax.annotate(
            '',
            xy=(center_x, bottom_y),
            xytext=(center_x, top_y),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=3
            )
        )
        ax.text(
            center_x + 50, (top_y + bottom_y) / 2,
            f'SCROLL{branch_idx+1}',
            color=color,
            fontsize=12,
            fontweight='bold',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor=color,
                boxstyle='round,pad=0.3'
            )
        )

    elif action_type == 'TYPE':
        # 显示输入文本
        text = params.get('text', '')
        screen_w, screen_h = screen_shape

        # 文本换行处理
        if len(text) > 20:
            text = text[:20] + '...'

        ax.text(
            screen_w / 2,
            screen_h / 2,
            f'TYPE: "{text}"',
            color='white',
            fontsize=14,
            ha='center',
            va='center',
            bbox=dict(
                facecolor=color,
                alpha=0.9,
                edgecolor='none',
                boxstyle='round,pad=0.5'
            )
        )

    elif action_type == 'COMPLETE':
        screen_w, screen_h = screen_shape
        ax.text(
            screen_w / 2,
            screen_h / 2,
            'COMPLETE',
            color='white',
            fontsize=16,
            ha='center',
            va='center',
            fontweight='bold',
            bbox=dict(
                facecolor=color,
                alpha=0.9,
                edgecolor='none',
                boxstyle='round,pad=0.5'
            )
        )

    elif action_type in ['HOME', 'BACK', 'ENTER', 'OPEN', 'IMPOSSIBLE']:
        screen_w, screen_h = screen_shape
        display_text = action_type
        if action_type == 'OPEN':
            display_text = f"OPEN: {params.get('app', 'Unknown')}"

        ax.text(
            screen_w / 2,
            screen_h / 2,
            display_text,
            color='white',
            fontsize=14,
            ha='center',
            va='center',
            bbox=dict(
                facecolor=ACTION_COLORS.get(action_type, 'gray'),
                alpha=0.9,
                edgecolor='none',
                boxstyle='round,pad=0.5'
            )
        )

    return next_state


def create_step_subplot(ax, state_key: str,
                        actions: List[Dict[str, Any]],
                        screenshot_path: Optional[str],
                        screen_shape: List[int],
                        show_flow: bool = True):
    """创建单步的子图"""
    # 加载并显示截图
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            image = Image.open(screenshot_path)
            img_array = np.array(image)
            ax.imshow(img_array)
            actual_shape = image.size
        except Exception as e:
            logger.warning(f"无法加载截图 {screenshot_path}: {e}")
            ax.set_facecolor('lightgray')
            actual_shape = screen_shape
    else:
        # 没有截图时使用纯色背景
        ax.set_facecolor('#E8E8E8')
        # 绘制占位文字
        ax.text(
            0.5, 0.5,
            f'No Screenshot\nState: {state_key}',
            transform=ax.transAxes,
            fontsize=14,
            ha='center',
            va='center',
            color='gray'
        )
        actual_shape = screen_shape

    # 绘制每个动作（支持分支）
    next_states = []
    for i, action in enumerate(actions):
        next_state = plot_action_on_axis(ax, action, actual_shape, branch_idx=i)
        if next_state:
            next_states.append(next_state)

    # 构建标题
    action_details = []
    for i, action in enumerate(actions):
        action_type = action.get('action', 'UNKNOWN')
        next_state = action.get('next', '')
        if len(actions) > 1:
            action_details.append(f"[{i+1}] {action_type}→{next_state}")
        else:
            action_details.append(f"{action_type}→{next_state}")

    title = f"Step {state_key}\n" + "\n".join(action_details)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')

    return next_states


def visualize_ref_data(data_dir: str, output_path: Optional[str] = None,
                       max_cols: int = 5, fig_width: int = 30,
                       show_flow: bool = True) -> bool:
    """
    可视化单个 ref.json 文件

    Args:
        data_dir: 包含 ref.json 和 screenshot 的目录
        output_path: 输出图片路径
        max_cols: 每行最多显示的步骤数
        fig_width: 图片宽度
        show_flow: 是否显示流转关系

    Returns:
        bool: 是否成功
    """
    ref_path = os.path.join(data_dir, 'ref.json')
    # 检查文件是否存在
    if not os.path.exists(ref_path):
        logger.error(f"找不到 ref.json: {ref_path}")
        return False

    # 加载数据
    ref_data = load_ref_json(ref_path)
    if not ref_data:
        return False

    # 提取元数据
    metadata = extract_metadata(ref_data)
    logger.info(f"处理: {data_dir}")
    logger.info(f"指令: {metadata['instruction']}")
    logger.info(f"应用: {metadata['app']}")

    # 提取状态
    states = extract_states(ref_data)
    if not states:
        logger.warning(f"没有找到任何状态节点: {ref_path}")
        return False

    state_order = get_state_order(states)
    logger.info(f"发现 {len(state_order)} 个状态: {state_order}")

    # 计算布局
    n_steps = len(state_order)
    n_cols = min(n_steps, max_cols)
    n_rows = (n_steps + n_cols - 1) // n_cols  # 向上取整

    # 创建图形
    row_height = fig_width * 0.6  # 根据宽度计算高度比例
    fig_height = row_height * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_steps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 绘制每个步骤
    all_next_states = {}
    for idx, state_key in enumerate(state_order):
        ax = axes[idx]
        actions = states[state_key]

        # 查找截图
        screenshot_path = find_screenshot(data_dir, state_key)

        # 创建子图
        next_states = create_step_subplot(
            ax, state_key, actions, screenshot_path,
            metadata['screen_shape'], show_flow
        )
        all_next_states[state_key] = next_states

    # 隐藏多余的子图
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')

    # 添加总标题
    fig.suptitle(
        f"[{metadata['app']}] {metadata['instruction']}",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    if output_path is None:
        output_path = os.path.join(data_dir, 'ref_visualization.png')

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    logger.info(f"可视化已保存: {output_path}")
    return True


def batch_visualize(data_dir: str, output_path: Optional[str] = None,
                    max_cols: int = 5, fig_width: int = 30,
                    show_flow: bool = True):
    """
    批量可视化目录下的所有 ref.json

    Args:
        data_dir: 包含多个子目录的根目录
        output_path: 输出图片路径
        max_cols: 每行最多显示的步骤数
        fig_width: 图片宽度
        show_flow: 是否显示流转关系
    """
    if not os.path.isdir(data_dir):
        logger.error(f"目录不存在: {data_dir}")
        return

    # 收集所有包含 ref.json 的子目录
    target_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            ref_path = os.path.join(item_path, 'ref.json')
            if os.path.exists(ref_path):
                target_dirs.append(item_path)

    if not target_dirs:
        logger.warning(f"在 {data_dir} 下没有找到任何包含 ref.json 的子目录")
        return

    logger.info(f"发现 {len(target_dirs)} 个任务目录")

    # 逐个处理
    success_count = 0
    for target_dir in sorted(target_dirs):
        # 如果指定了输出路径且是单个文件，为每个任务生成单独的文件
        if output_path and not os.path.isdir(output_path):
            base_name = os.path.basename(target_dir)
            dir_name = os.path.dirname(output_path)
            file_name = f"{base_name}_vis.png"
            out = os.path.join(dir_name, file_name) if dir_name else file_name
        else:
            out = None

        if visualize_ref_data(target_dir, out, max_cols, fig_width, show_flow):
            success_count += 1

    logger.info(f"批量可视化完成: {success_count}/{len(target_dirs)} 成功")


def main():
    """主函数"""
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir)

    if args.batch:
        # 批量模式
        batch_visualize(
            data_dir,
            args.output,
            args.max_cols,
            args.fig_width,
            args.show_flow
        )
    else:
        # 单目录模式
        output = args.output
        if output:
            output = os.path.abspath(output)

        success = visualize_ref_data(
            data_dir,
            output,
            args.max_cols,
            args.fig_width,
            args.show_flow
        )

        if not success:
            exit(1)


# ==========================================
#           TestVisualizer 类
# ==========================================

@dataclass
class StepRecord:
    """单步测试记录数据结构"""
    status: str
    screenshot: str
    action: str
    action_parameter: Dict[str, Any]
    raw_output: str
    ref_action: List[str]
    ref_params: List[Dict[str, Any]]
    check_result: bool


class TestVisualizer:
    """
    测试结果可视化器
    
    用于可视化 TestRunner 的测试结果，同时显示：
    - 合法动作范围（来自 ref.json）
    - Agent 实际操作
    - 操作正确性标记
    """
    
    def __init__(self, max_cols: int = 5, fig_width: int = 30):
        """
        初始化可视化器
        
        Args:
            max_cols: 每行最多显示的步骤数
            fig_width: 图片宽度
        """
        self.max_cols = max_cols
        self.fig_width = fig_width
    
    @staticmethod
    def convert_normalized_to_pixels(params: dict, width: int, height: int) -> dict:
        """
        将归一化坐标转换为实际像素坐标
        
        Args:
            params: 标准格式的参数字典（归一化坐标 [0, 1000]）
            width: 图像宽度
            height: 图像高度
            
        Returns:
            包含实际像素坐标的参数字典
        """
        result = {}
        for key, value in params.items():
            if key in ('point', 'start_point', 'end_point'):
                if isinstance(value, list) and len(value) >= 2:
                    x_norm = float(value[0])
                    y_norm = float(value[1])
                    x_pixel = int(x_norm / 1000 * width)
                    y_pixel = int(y_norm / 1000 * height)
                    result[key] = [x_pixel, y_pixel]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def plot_agent_click(self, ax, x: int, y: int, is_correct: bool, 
                         marker_size: int = 200):
        """
        绘制 Agent 点击操作
        
        Args:
            ax: matplotlib 坐标轴
            x: 点击 x 坐标（像素）
            y: 点击 y 坐标（像素）
            is_correct: 是否正确
            marker_size: 标记大小
        """
        color = AGENT_COLORS['correct'] if is_correct else AGENT_COLORS['incorrect']
        
        # 绘制空心圆形（边缘粗线条）
        radius = marker_size / 4
        circle = Circle(
            (x, y), 
            radius=radius,
            facecolor='none',
            edgecolor=color,
            linewidth=5,
            alpha=0.9,
            zorder=10
        )
        ax.add_patch(circle)
        
        # 在中心绘制 +号标记
        cross_size = radius * 0.5  # +号的臂长
        ax.plot([x - cross_size, x + cross_size], [y, y], 
                color=color, linewidth=3, zorder=11)
        ax.plot([x, x], [y - cross_size, y + cross_size], 
                color=color, linewidth=3, zorder=11)
    
    def plot_agent_scroll(self, ax, start: List[int], end: List[int],
                            is_correct: bool, arrow_width: int = 4):
        """
        绘制 Agent 滑动操作
        
        Args:
            ax: matplotlib 坐标轴
            start: 起点 [x, y]（像素）
            end: 终点 [x, y]（像素）
            is_correct: 是否正确
            arrow_width: 箭头宽度
        """
        color = AGENT_COLORS['correct'] if is_correct else AGENT_COLORS['incorrect']
        
        # 绘制箭头
        ax.annotate(
            '',
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=arrow_width,
                mutation_scale=20
            ),
            zorder=10
        )
        
        # 标记起点
        marker = '✓' if is_correct else '✗'
        ax.text(start[0], start[1] - 50, marker, color=color, 
               fontsize=16, ha='center', va='center', 
               fontweight='bold', zorder=11)
    
    def plot_agent_type(self, ax, text: str, is_correct: bool, 
                         screen_shape: Tuple[int, int]):
        """
        绘制 Agent 文本输入操作
        
        Args:
            ax: matplotlib 坐标轴
            text: 输入的文本
            is_correct: 是否正确
            screen_shape: 屏幕尺寸 (width, height)
        """
        color = AGENT_COLORS['correct'] if is_correct else AGENT_COLORS['incorrect']
        screen_w, screen_h = screen_shape
        
        # 截断长文本
        display_text = text[:25] + '...' if len(text) > 25 else text
        marker = '✓' if is_correct else '✗'
        
        ax.text(
            screen_w / 2, screen_h * 0.9,
            f'TYPE: "{display_text}" {marker}',
            color='white',
            fontsize=14,
            ha='center',
            va='center',
            fontweight='bold',
            bbox=dict(
                facecolor=color,
                alpha=0.9,
                edgecolor='white',
                boxstyle='round,pad=0.5'
            ),
            zorder=10
        )
    
    def plot_agent_open(self, ax, app_name: str, is_correct: bool,
                        screen_shape: Tuple[int, int]):
        """
        绘制 Agent 打开应用操作
        
        Args:
            ax: matplotlib 坐标轴
            app_name: 应用名称
            is_correct: 是否正确
            screen_shape: 屏幕尺寸 (width, height)
        """
        color = AGENT_COLORS['correct'] if is_correct else AGENT_COLORS['incorrect']
        screen_w, screen_h = screen_shape
        marker = '✓' if is_correct else '✗'
        
        ax.text(
            screen_w / 2, screen_h / 2,
            f'OPEN: {app_name} {marker}',
            color='white',
            fontsize=16,
            ha='center',
            va='center',
            fontweight='bold',
            bbox=dict(
                facecolor=color,
                alpha=0.9,
                edgecolor='white',
                boxstyle='round,pad=0.5'
            ),
            zorder=10
        )
    
    def plot_agent_complete(self, ax, is_correct: bool, 
                            screen_shape: Tuple[int, int]):
        """
        绘制 Agent 完成操作
        
        Args:
            ax: matplotlib 坐标轴
            is_correct: 是否正确
            screen_shape: 屏幕尺寸 (width, height)
        """
        color = AGENT_COLORS['correct'] if is_correct else AGENT_COLORS['incorrect']
        screen_w, screen_h = screen_shape
        marker = '✓' if is_correct else '✗'
        
        ax.text(
            screen_w / 2, screen_h / 2,
            f'COMPLETE {marker}',
            color='white',
            fontsize=18,
            ha='center',
            va='center',
            fontweight='bold',
            bbox=dict(
                facecolor=color,
                alpha=0.9,
                edgecolor='white',
                boxstyle='round,pad=0.5'
            ),
            zorder=10
        )
    
    def plot_agent_action(self, ax, action: str, params: dict, 
                          is_correct: bool, screen_shape: Tuple[int, int]):
        """
        绘制 Agent 操作（统一入口）
        
        Args:
            ax: matplotlib 坐标轴
            action: 动作类型
            params: 动作参数（像素坐标）
            is_correct: 是否正确
            screen_shape: 屏幕尺寸 (width, height)
        """
        if action == 'CLICK':
            point = params.get('point', [0, 0])
            if len(point) >= 2:
                self.plot_agent_click(ax, point[0], point[1], is_correct)
        
        elif action == 'SCROLL':
            start = params.get('start_point', [0, 0])
            end = params.get('end_point', [0, 0])
            if len(start) >= 2 and len(end) >= 2:
                self.plot_agent_scroll(ax, start, end, is_correct)
        
        elif action == 'TYPE':
            text = params.get('text', '')
            self.plot_agent_type(ax, text, is_correct, screen_shape)
        
        elif action == 'OPEN':
            app_name = params.get('app_name', 'Unknown')
            self.plot_agent_open(ax, app_name, is_correct, screen_shape)
        
        elif action == 'COMPLETE':
            self.plot_agent_complete(ax, is_correct, screen_shape)
    
    def plot_ref_action(self, ax, ref_action: str, ref_params: dict,
                        screen_shape: Tuple[int, int]):
        """
        绘制参考答案（合法范围）
        
        Args:
            ax: matplotlib 坐标轴
            ref_action: 参考动作
            ref_params: 参考参数
            screen_shape: 屏幕尺寸 (width, height)
        """
        color = 'cyan'  # 合法范围使用青色
        
        if ref_action == 'CLICK':
            # 优先使用 x_real/y_real（实际像素坐标），如果不存在则回退到 x/y（归一化坐标）
            x_range = ref_params.get('x_real', ref_params.get('x', [0, 100]))
            y_range = ref_params.get('y_real', ref_params.get('y', [0, 100]))
            
            rect = Rectangle(
                (x_range[0], y_range[0]),
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                linewidth=4,
                edgecolor=color,
                facecolor='none',
                alpha=0.8,
                linestyle='--'
            )
            ax.add_patch(rect)
            
            # 标注合法区域
            ax.text(
                x_range[0], y_range[1] + 20,
                'Valid Area',
                color=color,
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color)
            )
        
        elif ref_action == 'SCROLL':
            # 绘制参考滑动方向
            # 优先使用 x_real/y_real（实际像素坐标），如果不存在则回退到 x/y（归一化坐标）
            if 'x_real' in ref_params or 'x' in ref_params:
                x_range = ref_params.get('x_real', ref_params.get('x', []))
                y_range = ref_params.get('y_real', ref_params.get('y', []))
                if len(x_range) >= 2 and len(y_range) >= 2:
                    ax.annotate(
                        '',
                        xy=(x_range[1], y_range[1]),
                        xytext=(x_range[0], y_range[0]),
                        arrowprops=dict(
                            arrowstyle='->',
                            color=color,
                            lw=2,
                            linestyle='--'
                        ),
                        alpha=0.6
                    )
        
        elif ref_action == 'TYPE':
            text = ref_params.get('text', '')
            screen_w, screen_h = screen_shape
            ax.text(
                screen_w / 2, screen_h * 0.1,
                f'Expected: "{text[:20]}"',
                color=color,
                fontsize=12,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color)
            )
        
        elif ref_action == 'OPEN':
            app = ref_params.get('app', 'Unknown')
            screen_w, screen_h = screen_shape
            ax.text(
                screen_w / 2, screen_h * 0.1,
                f'Expected App: {app}',
                color=color,
                fontsize=12,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color)
            )
    
    def create_step_subplot(self, ax, step_record: Dict[str, Any],
                            screen_shape: Tuple[int, int] = (1080, 1920)):
        """
        创建单步可视化子图
        
        Args:
            ax: matplotlib 坐标轴
            step_record: 单步测试记录
            screen_shape: 屏幕尺寸 (width, height)
        """
        # 加载并显示截图
        screenshot_path = step_record.get('screenshot', '')
        actual_shape = screen_shape
        
        if screenshot_path and os.path.exists(screenshot_path):
            try:
                image = Image.open(screenshot_path)
                img_array = np.array(image)
                ax.imshow(img_array)
                actual_shape = image.size  # (width, height)
            except Exception as e:
                logger.warning(f"无法加载截图 {screenshot_path}: {e}")
                ax.set_facecolor('#E8E8E8')
        else:
            ax.set_facecolor('#E8E8E8')
            ax.text(0.5, 0.5, 'No Screenshot', transform=ax.transAxes,
                   fontsize=14, ha='center', va='center', color='gray')
        
        # 获取图像尺寸
        img_width, img_height = actual_shape
        
        # 绘制参考答案（合法范围）
        ref_actions = step_record.get('ref_action', [])
        ref_params = step_record.get('ref_params', [])
        
        if isinstance(ref_actions, list):
            for i, (ref_act, ref_par) in enumerate(zip(ref_actions, ref_params)):
                self.plot_ref_action(ax, ref_act, ref_par, actual_shape)
        elif isinstance(ref_actions, str):
            # 单个参考动作
            self.plot_ref_action(ax, ref_actions, ref_params[0] if ref_params else {}, 
                                actual_shape)
        
        # 绘制 Agent 实际操作
        agent_action = step_record.get('action', '')
        agent_params = step_record.get('action_parameter', {})
        is_correct = step_record.get('check_result', False)
        
        # 转换归一化坐标为像素坐标
        agent_params_pixel = self.convert_normalized_to_pixels(
            agent_params, img_width, img_height
        )
        
        if agent_action:
            self.plot_agent_action(ax, agent_action, agent_params_pixel, 
                                   is_correct, actual_shape)
        
        # 设置标题
        status = step_record.get('status', '?')
        result_text = 'PASS' if is_correct else 'FAIL'
        result_color = 'green' if is_correct else 'red'
        
        title = f"Step {status}: {agent_action}\n[{result_text}]"
        ax.set_title(title, fontsize=12, fontweight='bold', 
                    color=result_color, pad=10)
        ax.axis('off')
    
    def visualize_task(self, steps_record: List[Dict[str, Any]], 
                       output_dir: str,
                       instruction: str = '',
                       case_name: str = '') -> str:
        """
        可视化单个测试任务的所有步骤
        
        Args:
            steps_record: 步骤记录列表（来自 TestRunner.run_task 的返回值）
            output_dir: 输出目录
            instruction: 用户指令
            case_name: 用例名称
            
        Returns:
            汇总图片路径
        """
        if not steps_record:
            logger.warning("没有步骤记录，跳过可视化")
            return ''
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算布局
        n_steps = len(steps_record)
        n_cols = min(n_steps, self.max_cols)
        n_rows = (n_steps + n_cols - 1) // n_cols
        
        # 创建图形
        row_height = self.fig_width * 0.6
        fig_height = row_height * n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 figsize=(self.fig_width, fig_height))
        if n_steps == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # 绘制每个步骤
        for idx, step_record in enumerate(steps_record):
            ax = axes[idx]
            self.create_step_subplot(ax, step_record)
        
        # 隐藏多余的子图
        for idx in range(n_steps, len(axes)):
            axes[idx].axis('off')
        
        # 添加总标题
        if instruction:
            fig.suptitle(
                f"[{case_name}] {instruction}",
                fontsize=16,
                fontweight='bold',
                y=0.98
            )
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存汇总图片
        summary_path = os.path.join(output_dir, 'summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
        logger.info(f"可视化已保存: {summary_path}")
        return summary_path
    
    def visualize_single_step(self, step_record: Dict[str, Any],
                              output_path: str) -> str:
        """
        可视化单个步骤（左右对照）
        
        Args:
            step_record: 单步测试记录
            output_path: 输出图片路径
            
        Returns:
            图片路径
        """
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 10))

        # 左侧：我的结果（模型输出）；若未到达该步则留空
        has_prediction = bool(step_record.get("action"))
        if has_prediction:
            self.create_step_subplot(ax_left, step_record)
        else:
            ax_left.set_facecolor('#F5F7FA')
            ax_left.axis('off')
            ax_left.text(
                0.5, 0.5, "未到达该步",
                transform=ax_left.transAxes,
                ha='center', va='center',
                fontsize=16, color='#7F8C8D'
            )
        ax_left.set_title("我的结果（预测）", fontsize=12, fontweight='bold', color='#1f77b4', pad=10)

        # 右侧：官方答案（仅绘制参考动作，不绘制我的点击）
        right_record = dict(step_record)
        right_record["action"] = ""
        right_record["action_parameter"] = {}
        self.create_step_subplot(ax_right, right_record)
        status = step_record.get('status', '?')
        ax_right.set_title(f"官方答案（状态 {status} 可接受分支）", fontsize=12, fontweight='bold', color='#2ca02c', pad=10)

        # 底部文字：左边显示我的推理/动作，右边显示官方答案列表
        agent_action = step_record.get('action', '')
        agent_params = step_record.get('action_parameter', {})
        is_correct = step_record.get('check_result', False)
        raw_output = step_record.get('raw_output', '')
        ref_actions = step_record.get('ref_action', [])
        ref_params = step_record.get('ref_params', [])

        if has_prediction:
            left_text = (
                f"动作: {agent_action}\n"
                f"参数:\n{textwrap.fill(json.dumps(agent_params, ensure_ascii=False), width=55)}\n"
                f"是否命中: {'PASS' if is_correct else 'FAIL'}\n"
                "模型原始输出:\n"
                f"{textwrap.fill(str(raw_output)[:500], width=55)}"
            )
        else:
            left_text = "动作: (空)\n参数: {}\n是否命中: N/A\n模型原始输出: 未到达该步（提前失败或终止）"
        ref_pairs: List[str] = []
        if isinstance(ref_actions, list):
            for i, act in enumerate(ref_actions):
                params = ref_params[i] if isinstance(ref_params, list) and i < len(ref_params) else {}
                ref_pairs.append(f"{i + 1}. {act} {json.dumps(params, ensure_ascii=False)}")
        else:
            ref_pairs.append(f"1. {ref_actions} {json.dumps(ref_params, ensure_ascii=False)}")
        right_text = "官方可接受答案:\n" + "\n".join(ref_pairs[:8])

        fig.text(
            0.02, 0.02, left_text,
            ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='#eef6ff', alpha=0.85, edgecolor='#1f77b4')
        )
        fig.text(
            0.52, 0.02, right_text,
            ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='#eefbea', alpha=0.85, edgecolor='#2ca02c')
        )

        plt.tight_layout(rect=[0, 0.16, 1, 1])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
        return output_path


if __name__ == '__main__':
    # 独立运行时配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
