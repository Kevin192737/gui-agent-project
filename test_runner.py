"""
此文件提供测试评测功能。TestRunner 负责调用 Agent 进行测试并验证结果。

【重要提示】
提交阶段，该文件会被替换，所有修改都会被覆盖。
"""

import os
import json
import logging
import copy
import re
import csv
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from PIL import Image

from agent_base import (
    BaseAgent, AgentInput, AgentOutput, 
    DEFAULT_API_URL, DEFAULT_MODEL_ID, 
    TokenLimitExceeded, ConfigTamperError, UsageInfo,
    _is_production_mode
)

from utils.visualize_ref import TestVisualizer


# ==========================================
#               全局配置
# ==========================================

API_CONFIG = {
    "DATA_DIR": "./test_data/offline",
    "OUTPUT_DIR": "./output",
    "MAX_STEPS": 45,
    "MAX_TOTAL_TOKENS": 1200000  # Token 消耗总限制
}


# 确保输出目录存在
os.makedirs(API_CONFIG["OUTPUT_DIR"], exist_ok=True)

# 配置日志 - 清除已有 handlers 后重新配置，确保 FileHandler 生效
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{API_CONFIG["OUTPUT_DIR"]}/test_run.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)




# ==========================================
#               Checker 类
# ==========================================

class Checker:
    """验证器 - 验证标准格式的 Agent 输出
    """
    
    def __init__(self):
        # 标准动作到验证方法的映射
        self.check_map = {
            "CLICK": self._check_click,
            "SCROLL": self._check_scroll,
            "TYPE": self._check_type,
            "OPEN": self._check_open,
            "COMPLETE": self._check_no_params,
        }
        self.distance_threshold = 0.14
        self.angle_threshold = 10
    
    def get_screenshot(self, status: str, dir_path: str) -> Tuple[Image.Image, str]:
        """
        获取指定状态的截图
        
        Args:
            status: 状态标识（如 "0", "1-1"）
            dir_path: 测试用例目录路径
            
        Returns:
            (PIL Image, 截图文件路径)
        """
        pic_path = os.path.join(dir_path, status + '.png')
        if not os.path.exists(pic_path):
            pic_path = os.path.join(dir_path, status + '.jpg')
        
        if os.path.exists(pic_path):
            return Image.open(pic_path), pic_path
        else:
            logger.error(f"Screenshot not found: {pic_path}")
            return Image.new('RGB', (1080, 2400)), pic_path
    
    def calculate_distance(self, point1: list, point2: list) -> float:
        """计算两点之间的欧氏距离"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def calculate_angle(self, point1s: list, point1e: list, point2s: list, point2e: list) -> float:
        """计算两条线段之间的夹角（度数）"""
        p1s = np.array(point1s)
        p1e = np.array(point1e)
        p2s = np.array(point2s)
        p2e = np.array(point2e)
        vector1 = p1e - p1s
        vector2 = p2e - p2s
        dot = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        # 防止除零
        if norm1 == 0 or norm2 == 0:
            return 0
        cos_theta = dot / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        return np.degrees(theta_rad)
    
    def check(self, pred_action: str, pred_params: dict, 
             ref_action: str, ref_params: dict, 
             img_width: int, img_height: int) -> bool:
        """
        验证预测的动作和参数是否符合参考
        
        重要：此方法假设 Agent 输出已经符合标准格式，不做任何格式适配。
        
        Args:
            pred_action: 预测动作（标准格式，如 "CLICK"）
            pred_params: 预测参数（标准格式，如 {'point': [x, y]}），归一化坐标 [0, 1000]
            ref_action: 参考动作
            ref_params: 参考参数（ref.json 中的原始格式），包含归一化坐标 x/y 和像素坐标 x_real/y_real
            img_width: 图像宽度（保留参数，用于可能的扩展）
            img_height: 图像高度（保留参数，用于可能的扩展）
            
        Returns:
            是否匹配成功
        """
        # 直接比较动作类型（不再需要映射）
        if pred_action != ref_action:
            logger.info(f"[Checker] Action mismatch: expect [{ref_action}], got [{pred_action}]")
            return False
        
        # 获取验证方法
        check_method = self.check_map.get(pred_action)
        if not check_method:
            logger.error(f"[Checker] Unknown action type: {pred_action}")
            return False
        
        # 直接使用归一化坐标进行比较，无需转换
        # Agent 输出的坐标是归一化坐标 [0, 1000]
        # ref.json 中的 x/y 也是归一化坐标 [0, 1000]
        # 两者可以直接比较
        
        # 调用对应的验证方法
        try:
            return check_method(pred_params, ref_params)
        except Exception as e:
            logger.error(f"[Checker] Exception in check {pred_action}: {e}")
            return False
    
    def _check_click(self, pred_params: dict, ref_params: dict) -> bool:
        """
        验证 CLICK - 标准 params: {'point': [x, y]}（归一化坐标 [0, 1000]）
        
        Args:
            pred_params: Agent 输出的参数，归一化坐标
            ref_params: ref.json 中的参数，包含归一化坐标 x/y 和像素坐标 x_real/y_real
        """
        point = pred_params.get('point')
        if not point or len(point) != 2:
            logger.error(f"[Checker] Invalid point format: {point}")
            return False
        
        x, y = point
        
        # 使用归一化坐标进行比较（ref.json 中的 x/y 是归一化坐标）
        x_min, x_max = ref_params['x']
        y_min, y_max = ref_params['y']
        
        if x_min < x < x_max and y_min < y < y_max:
            logger.debug(f"[Checker] CLICK at ({x}, {y}), in ([{x_min}, {x_max}], [{y_min}, {y_max}])")
            return True
        else:
            logger.error(f"[Checker] CLICK failed: ({x}, {y}) not in ([{x_min}, {x_max}], [{y_min}, {y_max}])")
            return False
    
    def _check_scroll(self, pred_params: dict, ref_params: dict) -> bool:
        """
        验证 SCROLL - 标准 params: {'start_point': [x, y], 'end_point': [x, y]}（归一化坐标 [0, 1000]）
        
        Args:
            pred_params: Agent 输出的参数，归一化坐标
            ref_params: ref.json 中的参数
        """
        # 提取起点和终点
        start = pred_params.get('start_point')
        end = pred_params.get('end_point')
        
        if not start or not end or len(start) != 2 or len(end) != 2:
            logger.error(f"[Checker] Invalid SCROLL params: start={start}, end={end}")
            return False
        
        start_x, start_y = start
        end_x, end_y = end
        
        # ref.json 中的 SCROLL 可能没有 'x' 和 'y'（跳过检查）
        if "x" not in ref_params.keys():
            logger.info(f"[Checker] SCROLL: skip coordinate check")
            return True
        
        start_x_ref, end_x_ref = ref_params['x']
        start_y_ref, end_y_ref = ref_params['y']
        
        # 判断使用精确验证还是角度验证
        is_precise = ref_params.get('is_precise', None)
        
        if is_precise:
            # 精确验证：检查起点和终点是否在误差范围内
            if (self.calculate_distance([start_x, start_y], [start_x_ref, start_y_ref]) < self.distance_threshold and
                self.calculate_distance([end_x, end_y], [end_x_ref, end_y_ref]) < self.distance_threshold):
                return True
            else:
                logger.error(f"[Checker] SCROLL: distance NOT satisfy")
                return False
        elif is_precise is False:
            # 角度验证：检查滑动方向是否正确
            if self.calculate_angle([start_x, start_y], [end_x, end_y],
                                   [start_x_ref, start_y_ref], [end_x_ref, end_y_ref]) < self.angle_threshold:
                return True
            else:
                logger.error(f"[Checker] SCROLL: angle NOT satisfy")
                return False
        else:
            logger.info(f"[Checker] SCROLL: skip check (no is_precise specified)")
            return True
    
    def _check_type(self, pred_params: dict, ref_params: dict) -> bool:
        """验证 TYPE - 标准 params: {'text': '内容'}"""
        pred_text = pred_params.get('text', '')
        ref_text = ref_params['text']
        
        # 检查是否为正则表达式：
        # 1) 旧格式：'正则 xxx'
        # 2) 新格式：直接写模式，如 '.*国际医学中心'
        is_regex = False
        if '正则' in ref_text:
            ref_text = ref_text.replace('正则 ', '')
            is_regex = True
        elif any(token in ref_text for token in ['.*', '^', '$', '\\d', '\\w', '\\s', '[', ']', '(', ')', '|', '+', '?']):
            is_regex = True

        if is_regex:
            try:
                if re.match(ref_text, pred_text):
                    return True
                logger.error(f"[Checker] TYPE: regex mismatch, expect '{ref_text}', got '{pred_text}'")
                return False
            except re.error:
                # 如果模式非法，回退为精确匹配，避免校验崩溃
                pass

        if pred_text == ref_text:
            return True
        logger.error(f"[Checker] TYPE: text mismatch, expect '{ref_text}', got '{pred_text}'")
        return False
    
    def _check_open(self, pred_params: dict, ref_params: dict) -> bool:
        """验证 OPEN - 标准 params: {'app_name': '应用名'}"""
        pred_app = pred_params.get('app_name', '')
        ref_app = ref_params['app']
        
        if pred_app == ref_app:
            return True
        else:
            logger.error(f"[Checker] OPEN: app mismatch, expect '{ref_app}', got '{pred_app}'")
            return False
    
    def _check_no_params(self, pred_params: dict, ref_params: dict) -> bool:
        """验证无参数动作（COMPLETE）"""
        # 标准格式下，这些动作的 parameters 应该是空字典
        if pred_params:
            logger.warning(f"[Checker] Action should have no params, got {pred_params}")
        return True


# ==========================================
#               TestRunner 类
# ==========================================

class TestRunner:
    """测试评测器 - 选手不可修改"""
    
    def __init__(self, agent: BaseAgent, debug_test: bool = True):
        """
        初始化测试运行器
        
        Args:
            agent: Agent 实例（选手实现的）
            debug_test: 是否在验证失败时继续执行。True 为继续执行（适用于自测），
                       False 为终止执行（适用于打分阶段）。默认为 True。
        """
        # 校验 Agent 的 API 配置是否被篡改
        self._validate_agent_config(agent)
        self.agent = agent
        self.checker = Checker()
        self.max_steps = API_CONFIG.get("MAX_STEPS", 30)
        self.debug_test = debug_test
        
        # Token 消耗监控
        self._total_tokens = 0
        self._max_total_tokens = API_CONFIG.get("MAX_TOTAL_TOKENS", 100000)
        
        # 初始化可视化器
        self.visualizer = TestVisualizer()
    
    def _validate_agent_config(self, agent: BaseAgent):
        """验证 Agent 的 API 配置未被篡改
        
        采用双重验证机制：
        1. 配置签名验证：检查 Agent 初始化时的签名是否与当前属性一致
        2. 环境变量验证：提交阶段强制验证配置与主办方设置一致
        
        Args:
            agent: Agent 实例
            
        Raises:
            ConfigTamperError: 如果检测到配置被篡改
        """
        # 获取 Agent 实际返回的值
        actual_api_url = agent.api_url
        actual_model_id = agent.model_id
        
        # ==========================================
        # 第一重验证：配置签名验证
        # ==========================================
        # 计算当前属性的签名
        import hashlib
        current_data = f"{actual_api_url}|{actual_model_id}"
        current_signature = hashlib.md5(current_data.encode()).hexdigest()
        
        # 与初始化时的签名比较
        original_signature = agent.get_config_signature()
        if current_signature != original_signature:
            raise ConfigTamperError(
                f"检测到配置篡改！Agent 属性被动态修改。\n"
                f"原始签名: {original_signature}\n"
                f"当前签名: {current_signature}\n"
                f"评测已终止。"
            )
        
        # ==========================================
        # 第二重验证：环境变量验证（仅提交阶段）
        # ==========================================
        if _is_production_mode():
            # 提交阶段：强制验证配置与主办方设置一致
            expected_url = os.environ.get("EVAL_API_URL", DEFAULT_API_URL)
            expected_model = os.environ.get("EVAL_MODEL_ID", DEFAULT_MODEL_ID)
            
            if actual_api_url != expected_url:
                raise ConfigTamperError(
                    f"[提交阶段] API URL 配置错误！\n"
                    f"期望值: {expected_url}\n"
                    f"实际值: {actual_api_url}\n"
                    f"评测已终止。"
                )
            
            if actual_model_id != expected_model:
                raise ConfigTamperError(
                    f"[提交阶段] Model ID 配置错误！\n"
                    f"期望值: {expected_model}\n"
                    f"实际值: {actual_model_id}\n"
                    f"评测已终止。"
                )
            
            logger.info("[提交阶段] Agent 配置校验通过")
        else:
            # 调试阶段：显示警告提示选手
            logger.warning("")
            logger.warning("=" * 60)
            logger.warning("[调试模式] 当前 API 配置仅供自测使用")
            logger.warning(f"  API_URL:  {actual_api_url}")
            logger.warning(f"  MODEL_ID: {actual_model_id}")
            logger.warning("")
            logger.warning("提示：提交时将使用主办方统一配置")
            logger.warning("=" * 60)
            logger.warning("")
    
    def _check_token_limit(self, usage: UsageInfo) -> None:
        """
        检查 token 使用量是否超过限制
        
        Args:
            usage: Token 使用信息
            
        Raises:
            TokenLimitExceeded: 当累计 token 超过限制时抛出
        """
        if usage:
            self._total_tokens += usage.total_tokens
            logger.info(f"Token usage: +{usage.total_tokens} (input: {usage.input_tokens}, output: {usage.output_tokens}), total: {self._total_tokens}/{self._max_total_tokens}")
            
            if self._total_tokens > self._max_total_tokens:
                raise TokenLimitExceeded(self._total_tokens, self._max_total_tokens)
    
    @staticmethod
    def _encode_image_for_history(image: Image.Image, image_format: str = "PNG") -> str:
        """
        将图片编码为 base64 URL（用于历史消息）

        Args:
            image: PIL Image 对象
            image_format: 图片格式（默认 PNG）

        Returns:
            base64 编码的图片 URL 字符串
        """
        import io
        import base64
        buffered = io.BytesIO()
        image.save(buffered, format=image_format)
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{image_format.lower()};base64,{base64_str}"

    @staticmethod
    def _format_params(params: Dict[str, Any]) -> str:
        """
        将参数字典格式化为字符串

        Args:
            params: 参数字典

        Returns:
            格式化后的参数字符串
        """
        if not params:
            return ""
        param_strs = []
        for key, value in params.items():
            if isinstance(value, list):
                param_strs.append(f"{key}={value}")
            elif isinstance(value, str):
                param_strs.append(f"{key}='{value}'")
            else:
                param_strs.append(f"{key}={value}")
        return ", ".join(param_strs)

    def _transfer_ref_position(self, ref_params: dict, width: int, height: int) -> dict:
        """
        转换 ref.json 中的坐标参数为实际像素格式
        
        ref.json 中的坐标已经是像素坐标，此方法主要用于：
        1. 确保参数格式正确
        2. 可扩展用于未来的坐标转换需求
        
        Args:
            ref_params: ref.json 中的参数
            width: 图像宽度（未使用，保留用于扩展）
            height: 图像高度（未使用，保留用于扩展）
            
        Returns:
            转换后的参数字典
        """
        result = {}
        for key, value in ref_params.items():
            result[key] = value
        return result
    
    def run_task(self, screenshots_dir: str, visualization_dir: str) -> Dict[str, Any]:
        """
        执行单个测试任务

        Args:
            screenshots_dir: 测试用例目录路径
            visualization_dir: 可视化输出目录

        Returns:
            包含测试结果的字典，包含以下字段：
            - instruction: 用户指令
            - steps: 每一步的详细信息列表
            - current_status: 最终状态
            - next_status: 下一个状态
            - visualization_path: 可视化图片路径
        """
        # 重置 Agent 状态（每个测试用例开始前调用）
        try:
            self.agent.reset()
        except Exception as e:
            logger.warning(f"Agent reset failed: {e}")

        # 读取 ref.json
        ref_data = self._load_ref_data(screenshots_dir)
        
        # 获取初始状态和指令
        first_status, instruction, step_max = self._get_initial_info(ref_data)
        
        current_status = next_status = first_status
        step_count = 1
        steps_record = []
        
        # 初始化历史
        history_messages = []
        history_actions = []
        
        while next_status != '#' and step_count <= min(step_max, self.max_steps):
            current_status = next_status
            logger.info(f"--- Step {step_count}: Current Status {current_status} ---")
            
            # 1. 准备 AgentInput
            screenshot, screenshot_path = self.checker.get_screenshot(current_status, screenshots_dir)
            
            agent_input = AgentInput(
                instruction=instruction,
                current_image=screenshot,
                step_count=step_count,
                history_messages=history_messages,
                history_actions=history_actions
            )
            
            # 2. 调用 Agent（带每步验证）
            try:
                # 每步开始前验证配置
                self._validate_agent_config(self.agent)
                
                agent_output = self.agent.act(agent_input)
                logger.info(f"Agent Output: action={agent_output.action}, params={agent_output.parameters}")
                
                # 每步结束后验证配置（检测运行时篡改）
                self._validate_agent_config(self.agent)
                
                # 检查 token 使用量
                if agent_output.usage:
                    self._check_token_limit(agent_output.usage)
                    
            except TokenLimitExceeded:
                raise  # 向上传递 token 限制异常
            except Exception as e:
                logger.error(f"Agent execution error at step {step_count}: {e}")
                # 创建一个失败的输出
                agent_output = AgentOutput(
                    action="",
                    parameters={},
                    raw_output=f"Error: {str(e)}"
                )
            
            # 3. 验证结果（使用 Checker）
            is_ok, ref_action_list, ref_params_list, matched_next_status = self._check_result(
                ref_data, current_status, screenshot.width, screenshot.height,
                agent_output.action, agent_output.parameters
            )
            
            # 4. 更新历史（使用标准 OpenAI messages 格式）
            # 添加用户消息（包含当前截图）
            screenshot_base64 = self._encode_image_for_history(screenshot)
            history_messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": screenshot_base64}}
                ]
            })
            # 添加助手消息（包含动作）
            history_messages.append({
                "role": "assistant",
                "content": f"Action: {agent_output.action}({self._format_params(agent_output.parameters)})"
            })
            history_actions.append({
                "step": step_count,
                "action": agent_output.action,
                "parameters": agent_output.parameters,
                "raw_output": agent_output.raw_output,
                "is_valid": is_ok
            })
            
            # 5. 记录结果
            step_record = {
                "step": step_count,
                "status": current_status,
                "screenshot": screenshot_path,
                "action": agent_output.action,
                "action_parameter": agent_output.parameters,
                "raw_output": agent_output.raw_output,
                "ref_action": ref_action_list,
                "ref_params": ref_params_list,
                "check_result": is_ok,
                "matched_next_status": matched_next_status,
            }
            steps_record.append(step_record)
            
            # 6. 状态流转
            if is_ok and matched_next_status:
                next_status = matched_next_status
            else:
                # 如果错误，根据 debug_test 决定是否继续执行
                if not self.debug_test:
                    # 打分阶段：终止后续执行
                    logger.info(f"[TestRunner] 验证失败，终止后续执行 (debug_test=False)")
                    next_status = '#'
                else:
                    # 自测阶段：跳转到默认的下一个状态继续执行
                    possible_moves = ref_data.get(current_status, [])
                    if isinstance(possible_moves, list) and possible_moves:
                        next_status = possible_moves[0].get('next', '#')
                    elif isinstance(possible_moves, dict):
                        next_status = possible_moves.get('next', '#')
                    else:
                        next_status = '#'
            
            step_count += 1
        
        # 生成可视化
        visualization_path = ''
        if self.visualizer and steps_record:
            # 确定可视化输出目录
            if visualization_dir is None:
                visualization_dir = os.path.join(
                    API_CONFIG.get('OUTPUT_DIR', './output'), 
                    'visualization'
                )
            
            # 创建用例专属目录
            case_name = os.path.basename(screenshots_dir)
            case_vis_dir = os.path.join(visualization_dir, case_name)
            
            try:
                visualization_path = self.visualizer.visualize_task(
                    steps_record=steps_record,
                    output_dir=case_vis_dir,
                    instruction=instruction,
                    case_name=case_name
                )
            except Exception as e:
                logger.error(f"可视化生成失败: {e}")
                visualization_path = ''

        # 导出逐步对比表（每一步预测 vs 参考可接受答案）
        try:
            if visualization_dir is None:
                visualization_dir = os.path.join(
                    API_CONFIG.get('OUTPUT_DIR', './output'),
                    'visualization'
                )
            case_name = os.path.basename(screenshots_dir)
            case_vis_dir = os.path.join(visualization_dir, case_name)
            full_steps_record = self._build_full_steps_for_comparison(
                ref_data=ref_data,
                screenshots_dir=screenshots_dir,
                steps_record=steps_record,
            )
            self._save_step_comparison(full_steps_record, case_vis_dir)
            self._save_step_visualizations(full_steps_record, case_vis_dir)
        except Exception as e:
            logger.error(f"逐步对比导出失败: {e}")
        
        return {
            "instruction": instruction,
            "steps": steps_record,
            "current_status": current_status,
            "next_status": next_status,
            "visualization_path": visualization_path
        }

    def _save_step_comparison(self, steps_record: List[Dict[str, Any]], output_dir: str) -> str:
        """导出每一步预测结果与参考答案分支的对比 CSV。"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "step_comparison.csv")

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "status",
                    "pred_action",
                    "pred_params",
                    "ref_actions",
                    "ref_params",
                    "matched",
                    "matched_next_status",
                ],
            )
            writer.writeheader()
            for row in steps_record:
                writer.writerow(
                    {
                        "step": row.get("step", ""),
                        "status": row.get("status", ""),
                        "pred_action": row.get("action", ""),
                        "pred_params": json.dumps(row.get("action_parameter", {}), ensure_ascii=False),
                        "ref_actions": json.dumps(row.get("ref_action", []), ensure_ascii=False),
                        "ref_params": json.dumps(row.get("ref_params", []), ensure_ascii=False),
                        "matched": bool(row.get("check_result", False)),
                        "matched_next_status": row.get("matched_next_status", ""),
                    }
                )
        return output_path

    def _save_step_visualizations(self, steps_record: List[Dict[str, Any]], output_dir: str) -> None:
        """导出每一步可视化对比图。"""
        if not self.visualizer:
            return
        os.makedirs(output_dir, exist_ok=True)
        for row in steps_record:
            step_idx = int(row.get("step", 0))
            output_path = os.path.join(output_dir, f"step_{step_idx:02d}_compare.png")
            self.visualizer.visualize_single_step(row, output_path)

    def _build_full_steps_for_comparison(
        self,
        ref_data: Dict[str, Any],
        screenshots_dir: str,
        steps_record: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        构造完整步骤列表：
        - 已执行步骤：保留原记录
        - 未到达步骤：补空白预测，只展示官方答案（用于可视化对比）
        """
        full_steps = list(steps_record)
        existing_statuses = {str(item.get("status", "")) for item in steps_record}
        next_step = len(steps_record) + 1

        numeric_statuses = sorted([k for k in ref_data.keys() if str(k).isdigit()], key=lambda x: int(x))
        for status in numeric_statuses:
            if status in existing_statuses:
                continue

            possible_moves = ref_data.get(status, [])
            if not isinstance(possible_moves, list):
                possible_moves = [possible_moves]

            ref_action_list = []
            ref_params_list = []
            for move in possible_moves:
                if not isinstance(move, dict):
                    continue
                ref_action_list.append(move.get("action", ""))
                ref_params_list.append(move.get("params", {}))

            screenshot_path_png = os.path.join(screenshots_dir, f"{status}.png")
            screenshot_path_jpg = os.path.join(screenshots_dir, f"{status}.jpg")
            screenshot_path = screenshot_path_png if os.path.exists(screenshot_path_png) else screenshot_path_jpg

            full_steps.append(
                {
                    "step": next_step,
                    "status": status,
                    "screenshot": screenshot_path,
                    "action": "",
                    "action_parameter": {},
                    "raw_output": "未到达该步（提前失败或终止）",
                    "ref_action": ref_action_list,
                    "ref_params": ref_params_list,
                    "check_result": False,
                    "matched_next_status": "",
                }
            )
            next_step += 1

        return full_steps
    
    def _load_ref_data(self, screenshots_dir: str) -> Dict[str, Any]:
        """加载 ref.json"""
        ref_data_file = os.path.join(screenshots_dir, "ref.json")
        try:
            with open(ref_data_file, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ref.json: {e}")
            return {}
    
    def _get_initial_info(self, ref_data: Dict[str, Any]) -> Tuple[str, str, int]:
        """获取初始状态、指令和最大步数"""
        # 从 case_overview 获取指令信息
        case_overview = ref_data.get('case_overview', {})
        instruction = case_overview.get('instruction', 'No instruction found')
        
        # 计算最大步数：统计数字键的数量
        step_keys = [k for k in ref_data.keys() if k.isdigit()]
        step_max = len(step_keys) if step_keys else 20
        
        # 确定初始状态
        if '0-0' in ref_data:
            first_status = '0'
            if not os.path.exists(os.path.join('.', '0.png')):  # 这里只是判断命名格式
                first_status = '0-0'
        else:
            first_status = '1-1' if '1-1' in ref_data else '0'
        
        return first_status, instruction, step_max
    
    def _check_result(
        self,
        ref_data: Dict[str, Any],
        current_status: str,
        width: int,
        height: int,
        action: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, List, List, str]:
        """
        检查 Agent 输出是否符合参考
        
        Returns:
            (is_ok, ref_action_list, ref_params_list, matched_next_status)
        """
        is_ok = False
        matched_next_status = None
        
        # 获取当前状态的所有可能正确路径
        possible_moves = ref_data.get(current_status, [])
        if not isinstance(possible_moves, list):
            possible_moves = [possible_moves]
        
        ref_action_list = []
        ref_params_list = []
        
        for move in possible_moves:
            if not isinstance(move, dict):
                continue
            
            ref_action = move.get('action', '')
            raw_ref_params = move.get('params', {})
            # 转换坐标为真实像素
            real_ref_params = self._transfer_ref_position(raw_ref_params, width, height)
            
            ref_action_list.append(ref_action)
            ref_params_list.append(real_ref_params)
            
            # 检查是否匹配（传入图像尺寸用于转换归一化坐标）
            if self.checker.check(action, params, ref_action, real_ref_params, width, height):
                is_ok = True
                matched_next_status = move.get('next')
                break
        
        return is_ok, ref_action_list, ref_params_list, matched_next_status

    def run_all_tasks(
        self,
        data_dir: str = None,
        output_dir: str = None,
        case_filter: str = None,
    ) -> Dict[str, Any]:
        """
        运行所有测试任务并生成统计报告
        
        Args:
            data_dir: 测试数据目录，默认使用 API_CONFIG 中的设置
            output_dir: 结果输出目录，默认使用 API_CONFIG 中的设置
            case_filter: 若设置，仅运行该用例目录名（basename 或相对 data_dir 的路径）
            
        Returns:
            统计结果字典
        """
        if data_dir is None:
            data_dir = API_CONFIG['DATA_DIR']
        if output_dir is None:
            output_dir = API_CONFIG['OUTPUT_DIR']
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} does not exist. Please create it and add test cases.")
            return {}
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 可视化输出目录
        visualization_dir = os.path.join(output_dir, 'visualization')
        
        # 收集所有测试用例
        test_case_paths = []
        for root, dirs, files in os.walk(data_dir):
            if 'ref.json' in files:
                test_case_paths.append(root)
        
        test_case_paths.sort()

        if case_filter:
            cf = case_filter.replace("\\", "/").strip().strip("/")
            filtered: List[str] = []
            for p in test_case_paths:
                rel = os.path.relpath(p, data_dir).replace("\\", "/")
                base = os.path.basename(p)
                if base == cf or rel == cf or rel.endswith("/" + cf):
                    filtered.append(p)
            test_case_paths = filtered
            if not test_case_paths:
                logger.error("case_filter=%r 未匹配到任何用例（data_dir=%s）", case_filter, data_dir)
                return {}
        
        # 统计变量
        sum_count = 0
        ok_count = 0
        all_step_pass_count = 0
        all_step_count = 0
        ok_list = []
        error_list = []
        
        # 结果收集
        result_data = {
            "用例名": [],
            "用户指令": [],
            "测试结果": [],
            "PASS步数": [],
            "总步数": [],
            "应用名": []
        }
        
        # 遍历所有测试用例
        for path in test_case_paths:
            item = os.path.relpath(path, data_dir)
            
            # 执行测试
            logger.info(f"Start testing case: {item}")
            try:
                result = self.run_task(path, visualization_dir)
            except TokenLimitExceeded as e:
                logger.error(f"Token limit exceeded: {e}")
                logger.error(f"评测终止。累计 token: {self._total_tokens}, 限制: {self._max_total_tokens}")
                break
            
            sum_count += 1
            steps = result["steps"]
            step_total = len(steps)
            step_pass = sum(1 for step in steps if step["check_result"])
            
            # debug_test 模式下才计算步骤准确率（no_debug_test 模式下步骤统计无意义）
            if self.debug_test:
                step_score = step_pass / step_total if step_total > 0 else 0
                all_step_pass_count += step_pass
                all_step_count += step_total
            
            whole_score = 1 if (step_pass == step_total and step_total > 0) else 0
            
            logger.info('---------------------------------------------------')
            if self.debug_test:
                logger.info(f'[No. {sum_count}: {item}] Score: {step_pass}/{step_total} = {step_score:.2f}')
            else:
                logger.info(f'[No. {sum_count}: {item}] Result: {"PASS" if whole_score == 1 else "FAIL"}')
            logger.info('---------------------------------------------------')
            
            if whole_score == 1:
                ok_list.append(item)
            else:
                error_list.append(item)
            
            # 收集数据
            result_data["用例名"].append(item)
            result_data["用户指令"].append(result["instruction"])
            result_data["测试结果"].append(whole_score)
            result_data["PASS步数"].append(step_pass)
            result_data["总步数"].append(step_total)
            
            # 简单的应用名推断
            app_name = "Unknown"
            if "dazhongdianping" in item:
                app_name = "大众点评"
            elif "douyin" in item:
                app_name = "抖音"
            elif "jingdong" in item:
                app_name = "京东"
            elif "meituan" in item:
                app_name = "美团"
            elif "pinduoduo" in item:
                app_name = "拼多多"
            elif "taobao" in item:
                app_name = "淘宝"
            elif "12306" in item:
                app_name = "铁路12306"
            result_data["应用名"].append(app_name)
        
        # 最终统计
        logger.info('===============================================')
        logger.info("PASS List: " + str(ok_list))
        logger.info("FAIL List: " + str(error_list))
        logger.info('===============================================')
        
        total_whole_score = len(ok_list) / sum_count if sum_count > 0 else 0
        
        if self.debug_test:
            total_step_score = all_step_pass_count / all_step_count if all_step_count > 0 else 0
            logger.info(f'Step Level Accuracy: {total_step_score:.2f}')
            logger.info(f'Case Level Accuracy: {total_whole_score:.2f}')
        else:
            logger.info(f'Case Level Accuracy: {total_whole_score:.2f}')
            logger.info('(Step Level Accuracy 已跳过，no_debug_test 模式下步骤统计无意义)')
        
        # 保存结果到 Excel
        try:
            df = pd.DataFrame(result_data)
            out_file = os.path.join(output_dir, 'result.xlsx')
            df.to_excel(out_file, index=False)
            logger.info(f"Result saved to {out_file}")
        except Exception as e:
            logger.error(f"Failed to save excel: {e}")
        
        # 构建返回结果
        result = {
            "total_cases": sum_count,
            "passed_cases": len(ok_list),
            "case_accuracy": total_whole_score,
            "pass_list": ok_list,
            "fail_list": error_list
        }
        
        # debug_test 模式下才返回步骤准确率相关字段
        if self.debug_test:
            total_step_score = all_step_pass_count / all_step_count if all_step_count > 0 else 0
            result["step_accuracy"] = total_step_score
            result["total_steps"] = all_step_count
            result["passed_steps"] = all_step_pass_count
        
        return result


# ==========================================
#               主程序入口
# ==========================================

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(
        description='GUI Agent 测试评测框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_runner.py                           # 默认运行，debug_test=True，继续执行
  python test_runner.py --data_dir ./test_data    # 指定测试数据目录
  python test_runner.py --output_dir ./output     # 指定输出目录
  python test_runner.py --no_debug_test           # 打分模式，验证失败时终止执行
  python test_runner.py -c step_aiqiyi_onekey_0011 # 只跑一个用例（相对 offline 目录名）
        """
    )
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default=None,
        help='测试数据目录（默认: ./test_data/offline）'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='结果输出目录（默认: ./output）'
    )
    parser.add_argument(
        '--no_debug_test',
        action='store_true',
        help='打分模式：验证失败时终止后续执行（默认为自测模式，继续执行）'
    )
    parser.add_argument(
        '--case', '-c',
        type=str,
        default=None,
        help='只运行指定用例子目录名（相对 data_dir），如 step_aiqiyi_onekey_0011',
    )
    return parser.parse_args()


if __name__ == '__main__':
    """主程序入口 - 用于测试 TestRunner"""
    args = parse_args()
    
    from agent import Agent
    
    # 创建 Agent 实例
    agent = Agent()
    
    # debug_test 参数：默认为 True（自测模式），使用 --no_debug_test 时为 False（打分模式）
    debug_test = not args.no_debug_test
    
    # 创建测试运行器
    runner = TestRunner(agent, debug_test=debug_test)
    
    # 运行所有测试任务
    results = runner.run_all_tasks(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        case_filter=args.case,
    )
    
    if results:
        print("\n" + "="*50)
        print("测试完成!")
        print(f"总用例数: {results['total_cases']}")
        print(f"通过用例数: {results['passed_cases']}")
        print(f"用例准确率: {results['case_accuracy']:.2%}")
        if debug_test:
            print(f"步骤准确率: {results['step_accuracy']:.2%}")
        print("="*50)

