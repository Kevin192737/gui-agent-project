"""
Agent 基类和接口定义

选手通过继承 BaseAgent 类来实现自己的 Agent。
此文件定义了 Agent 的输入输出数据结构和基类接口。

【重要提示】
提交阶段，该文件会被替换，所有修改都会被覆盖。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from PIL import Image
import io
import os
import base64
import logging
import hashlib
import warnings

logger = logging.getLogger(__name__)


# ==========================================
#               API 调用保护相关
# ==========================================

# 禁止在 kwargs 中传入的敏感参数列表
FORBIDDEN_KWARGS = {
    "base_url", "api_key", "model", "model_id",
    "base_url", "endpoint", "url", "host"
}


# ==========================================
#               Token 限制异常
# ==========================================

class TokenLimitExceeded(Exception):
    """Token 使用量超过限制异常"""
    def __init__(self, current_tokens: int, limit: int):
        self.current_tokens = current_tokens
        self.limit = limit
        super().__init__(
            f"Token limit exceeded: {current_tokens} > {limit}"
        )


class ConfigTamperError(Exception):
    """配置篡改异常"""
    pass


# ==========================================
#            选手调试配置说明
# ==========================================
# 调试阶段可设置以下环境变量自定义配置：
#   DEBUG_API_URL  - API 地址（可选，默认使用下方 DEFAULT_API_URL）
#   DEBUG_MODEL_ID - 模型 ID（可选，默认使用下方 DEFAULT_MODEL_ID）
#   VLM_API_KEY    - 你的 API 密钥（必须设置）
#
# !!! 重要提示 !!!
# 1. 提交时，DEBUG_* 环境变量将被忽略
# 2. 主办方将使用统一的 EVAL_* 配置
# 3. 任何尝试篡改配置的行为都将被检测并终止评测
# ==========================================


# ==========================================
#               固定 API 配置
# ==========================================
# 模块级常量
DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL_ID = "doubao-seed-1-6-flash-250828"



def _is_production_mode() -> bool:
    """检查是否为提交阶段（生产模式）"""
    return os.environ.get("EVAL_MODE", "").lower() == "production"


def _get_api_url() -> str:
    """
    获取 API URL
    
    - 提交阶段：强制使用 EVAL_API_URL
    - 调试阶段：允许使用 DEBUG_API_URL，默认使用 DEFAULT_API_URL
    """
    if _is_production_mode():
        return os.environ.get("EVAL_API_URL", DEFAULT_API_URL)
    else:
        debug_url = os.environ.get("DEBUG_API_URL")
        if debug_url:
            warnings.warn(
                f"\n{'='*60}\n"
                f"[调试模式] 使用自定义 API_URL: {debug_url}\n"
                f"注意：提交时此配置将被替换为主办方统一配置\n"
                f"{'='*60}",
                UserWarning
            )
        return debug_url if debug_url else DEFAULT_API_URL


def _get_model_id() -> str:
    """
    获取模型 ID
    
    - 提交阶段：强制使用 EVAL_MODEL_ID
    - 调试阶段：优先使用 DEBUG_MODEL_ID，否则使用 DEFAULT_MODEL_ID
    """
    if _is_production_mode():
        return os.environ.get("EVAL_MODEL_ID", DEFAULT_MODEL_ID)
    else:
        debug_model = os.environ.get("DEBUG_MODEL_ID")
        if debug_model:
            warnings.warn(
                f"\n{'='*60}\n"
                f"[调试模式] 使用自定义 MODEL_ID: {debug_model}\n"
                f"注意：提交时此配置将被替换为主办方统一配置\n"
                f"{'='*60}",
                UserWarning
            )
        return debug_model if debug_model else DEFAULT_MODEL_ID


def _get_api_key() -> str:
    """
    获取 API Key
    
    - 提交阶段：强制使用 EVAL_API_KEY
    - 调试阶段：使用 VLM_API_KEY
    """
    if _is_production_mode():
        return os.environ.get("EVAL_API_KEY", "")
    else:
        return os.environ.get("VLM_API_KEY", "")


# ==========================================
#               标准动作常量
# ==========================================
# Agent 返回的动作必须是以下常量之一

ACTION_CLICK = "CLICK"
ACTION_SCROLL = "SCROLL"
ACTION_TYPE = "TYPE"
ACTION_OPEN = "OPEN"
ACTION_COMPLETE = "COMPLETE"

# 所有有效动作的集合
VALID_ACTIONS = {
    ACTION_CLICK,
    ACTION_SCROLL,
    ACTION_TYPE,
    ACTION_OPEN,
    ACTION_COMPLETE,
}


# ==========================================
#               Token 使用信息
# ==========================================

@dataclass
class UsageInfo:
    """Token 使用信息
    
    用于记录单次 LLM 调用的 token 消耗情况。
    
    Attributes:
        input_tokens: 输入 token 数量
        output_tokens: 输出 token 数量
        total_tokens: 总 token 数量
        cached_tokens: 缓存命中的 token 数量（可选）
        reasoning_tokens: 推理 token 数量（可选）
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


# ==========================================
#               标准参数格式
# ==========================================
# 坐标必须使用归一化坐标：0-1000

"""
标准参数格式要求（Agent 必须严格遵守）：

1. CLICK:
   {"point": [x, y]}
   - point: 归一化坐标 [x, y]，范围为 [0, 1000]

2. SCROLL:
   {"start_point": [x, y], "end_point": [x, y]}
   - start_point: 起始归一化坐标 [x, y]
   - end_point: 结束归一化坐标 [x, y]

3. TYPE:
   {"text": "内容"}
   - text: 输入的文本内容

4. OPEN:
   {"app_name": "应用名"}
   - app_name: 应用名称字符串

5. COMPLETE:
   {}
   - 空字典，无需参数

注意：
- 所有坐标必须是归一化的 [0, 1000] 范围
- 参数字典的 key 必须与上述标准一致
- TestRunner 不会进行任何格式适配，不符合标准格式将直接判定为失败
"""


@dataclass
class AgentInput:
    """Agent 输入数据结构 - 支持多种参数，选手可选择性使用
    
    Attributes:
        instruction: 用户原始指令
        current_image: 当前轮的截图
        step_count: 当前是第几次调用
        history_messages: 历史消息列表（选手可自定义格式）
        history_actions: 历史动作列表（选手可自定义格式）
        extra: 选手自定义扩展参数
    """
    instruction: str
    current_image: Image.Image
    step_count: int
    history_messages: List[Dict[str, Any]] = field(default_factory=list)
    history_actions: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Agent 输出数据结构 - 必须遵循严格标准格式

    重要：Agent 必须确保输出符合标准格式，TestRunner 不会进行任何格式适配。

    Attributes:
        action: 动作名称，必须是标准动作常量之一（ACTION_CLICK, ACTION_SCROLL 等）
        parameters: 动作参数，必须符合标准参数格式要求
        raw_output: 原始模型输出（可选，仅用于调试和日志记录）
        usage: Token 使用信息（可选，用于 token 消耗监控）

    标准格式示例：
        action=ACTION_CLICK, parameters={"point": [500, 300]}
        action=ACTION_SCROLL, parameters={"start_point": [500, 500], "end_point": [500, 900]}
        action=ACTION_TYPE, parameters={"text": "搜索"}
        action=ACTION_OPEN, parameters={"app_name": "淘宝"}
        action=ACTION_COMPLETE, parameters={}

    错误示例（这些不会被 TestRunner 接受）：
        action="click"  # 错误：必须是大写的常量
        action=ACTION_CLICK, parameters={"coord": [500, 300]}  # 错误：key 必须是 "point"
        action=ACTION_CLICK, parameters=[500, 300]  # 错误：必须是字典
    """
    action: str
    parameters: Dict[str, Any]
    raw_output: str = ""
    usage: Optional[UsageInfo] = None


class BaseAgent:
    """Agent 基类 - 选手继承此类实现自己的 Agent

    选手需要：
    1. 继承 BaseAgent
    2. 实现 act() 方法
    3. 可选重写 generate_messages() 方法以自定义消息生成逻辑
    4. 可选重写 _initialize() 方法进行初始化

    重要：API URL 和 Model ID 已固定，选手不可修改。仅需配置 VLM_API_KEY 环境变量。
    
    调试阶段说明：
    - 可通过 DEBUG_API_URL、DEBUG_MODEL_ID 环境变量自定义配置
    - 提交阶段这些配置将被忽略，强制使用主办方统一配置
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 Agent

        Args:
            config: 配置字典（保留用于其他扩展配置）

        环境变量：
            调试阶段：
                DEBUG_API_URL: 自定义 API 地址（可选）
                DEBUG_MODEL_ID: 自定义模型 ID（可选）
                VLM_API_KEY: API 密钥（必须设置）
            
            提交阶段（由主办方设置）：
                EVAL_MODE: 设置为 "production"
                EVAL_API_URL: 统一的 API 地址
                EVAL_MODEL_ID: 统一的模型 ID
                EVAL_API_KEY: 统一的 API 密钥
        """
        self.config = config or {}
        
        # 存储配置到私有变量（用于签名验证）
        self._api_url = _get_api_url()
        self._model_id = _get_model_id()
        self._api_key = _get_api_key()
        
        # 计算配置签名，用于防篡改验证
        self._config_signature = self._compute_config_signature()
        
        self._initialize()

    def _compute_config_signature(self) -> str:
        """
        计算配置签名，用于防止篡改
        
        签名基于实际的配置值，子类如果重写属性会改变实际值，
        TestRunner 可以通过比较签名来检测篡改。
        """
        data = f"{self._api_url}|{self._model_id}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_config_signature(self) -> str:
        """
        获取配置签名（供 TestRunner 验证使用）
        
        返回初始化时的配置签名，如果子类通过重写属性篡改配置，
        实际返回值将与签名不一致，从而被检测到。
        """
        return self._config_signature

    @property
    def api_url(self) -> str:
        """
        API URL - 只读属性
        
        返回值在初始化时确定，提交阶段使用主办方统一配置，
        调试阶段可使用 DEBUG_API_URL 自定义。
        """
        return self._api_url

    @property
    def model_id(self) -> str:
        """
        Model ID - 只读属性
        
        返回值在初始化时确定，提交阶段使用主办方统一配置，
        调试阶段可使用 DEBUG_MODEL_ID 自定义。
        """
        return self._model_id

    @property
    def api_key(self) -> str:
        """
        API Key - 只读属性
        
        返回值在初始化时确定，提交阶段使用主办方统一配置，
        调试阶段使用 VLM_API_KEY。
        """
        return self._api_key
    
    def _initialize(self):
        """初始化方法，子类可重写
        
        用于初始化模型客户端、加载配置等
        """
        pass
    
    def generate_messages(
        self, 
        input_data: AgentInput
    ) -> List[Dict[str, Any]]:
        """
        生成发给大模型的 messages
        
        默认实现：简单的 system prompt + user instruction + current image
        子类可重写以支持历史消息、多轮对话等
        
        Args:
            input_data: AgentInput 包含所有可用信息
            
        Returns:
            符合 OpenAI 格式的 messages 列表
        """
        # 默认简单实现
        system_prompt = self._build_system_prompt(input_data.instruction)
        
        messages = [
            {"role": "user", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": self._encode_image(input_data.current_image)}}]
            }
        ]
        
        return messages
    
    def _build_system_prompt(self, instruction: str) -> str:
        """
        构建系统提示词，子类可重写
        
        Args:
            instruction: 用户指令
            
        Returns:
            系统提示词字符串
        """
        return f"""You are a GUI agent. You need to complete the following task.

## Task
{instruction}

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(point='<point>x y</point>')
type(content='')
scroll(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
open(app_name='')
complete(content='xxx')

## Note
- Use Chinese in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
"""
    
    def _encode_image(self, image: Image.Image, image_format: str = "PNG") -> str:
        """
        将图片编码为 base64 URL
        
        Args:
            image: PIL Image 对象
            image_format: 图片格式（默认 PNG）
            
        Returns:
            base64 编码的图片 URL 字符串
        """
        buffered = io.BytesIO()
        image.save(buffered, format=image_format)
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{image_format.lower()};base64,{base64_str}"
    
    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent 核心方法：根据输入生成动作

        子类必须实现此方法，内部可以：
        1. 调用 generate_messages 生成 messages
        2. 调用大模型
        3. 解析输出为 action 和 parameters
        4. 返回 AgentOutput

        Args:
            input_data: AgentInput 包含当前轮所有信息

        Returns:
            AgentOutput 包含动作和参数
        """
        raise NotImplementedError("Subclass must implement act method")

    def reset(self):
        """
        重置 Agent 状态

        在每个测试用例开始前由 TestRunner 调用，用于重置 Agent 的内部状态。
        子类应重写此方法以清空历史消息、计划等状态。

        示例：
            def reset(self):
                self._message_history = []
                self._task_plan = []
                self._current_plan_step = 0
        """
        pass
    
    def _call_api(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """
        受保护的 API 调用方法
        
        选手必须通过此方法调用大模型 API，禁止自行创建客户端或绕过此方法。
        此方法确保 API 调用使用固定的配置（api_url, model_id, api_key）。
        
        Args:
            messages: 符合 OpenAI 格式的消息列表
            **kwargs: 额外的 API 调用参数（如 temperature, top_p 等）
                     注意：base_url、api_key、model 等敏感参数会被强制移除
        
        Returns:
            API 响应对象（OpenAI ChatCompletion 格式）
        
        Raises:
            ConfigTamperError: 如果检测到篡改尝试（如在 kwargs 中传入敏感参数）
        
        使用示例：
            def act(self, input_data: AgentInput) -> AgentOutput:
                messages = self.generate_messages(input_data)
                response = self._call_api(messages, temperature=0, top_p=0.7)
                content = response.choices[0].message.content
                # 解析 content 并返回 AgentOutput...
        """
        # 1. 检查并移除敏感参数
        forbidden_found = []
        
        for key, value in kwargs.items():
            if key.lower() in FORBIDDEN_KWARGS or key in FORBIDDEN_KWARGS:
                forbidden_found.append(key)
        
        if forbidden_found:
            logger.warning(
                f"[安全警告] 以下敏感参数已被移除: {forbidden_found}。"
                f"请勿尝试传入 base_url、api_key、model 等参数。"
            )
        
        # 2. 运行时签名验证（防止运行时篡改私有变量）
        current_signature = self._compute_runtime_signature()
        if current_signature != self._config_signature:
            raise ConfigTamperError(
                f"检测到配置篡改！运行时签名与初始化签名不一致。\n"
                f"初始签名: {self._config_signature}\n"
                f"当前签名: {current_signature}\n"
                f"评测已终止。"
            )
        
        # 3. 延迟导入 OpenAI（避免未安装时报错）
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai 包: pip install openai"
            )
        
        # 4. 创建临时客户端并调用 API
        client = OpenAI(
            base_url=self._api_url,
            api_key=self._api_key
        )
        
        # 记录 API 调用日志
        logger.info(f"[API调用] model={self._model_id}, url={self._api_url}")
        
        # 调用 API
        completion = client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )
        
        return completion
    
    def _compute_runtime_signature(self) -> str:
        """
        计算运行时签名（用于运行时验证）
        
        与 _compute_config_signature 不同，此方法直接读取私有变量，
        而不是使用可能被重写的属性。
        """
        data = f"{self._api_url}|{self._model_id}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def extract_usage_info(self, response: Any) -> UsageInfo:
        """
        从 API 响应中提取 Token 使用信息
        
        Args:
            response: OpenAI API 响应对象
            
        Returns:
            UsageInfo 实例
        """
        usage = UsageInfo()
        
        if hasattr(response, 'usage') and response.usage:
            usage.input_tokens = (
                getattr(response.usage, 'prompt_tokens', 0) or 
                getattr(response.usage, 'input_tokens', 0)
            )
            usage.output_tokens = (
                getattr(response.usage, 'completion_tokens', 0) or 
                getattr(response.usage, 'output_tokens', 0)
            )
            usage.total_tokens = getattr(response.usage, 'total_tokens', 0)
            
            # 尝试提取更详细的信息
            details = (
                getattr(response.usage, 'prompt_tokens_details', None) or 
                getattr(response.usage, 'input_tokens_details', None)
            )
            if details and hasattr(details, 'cached_tokens'):
                usage.cached_tokens = details.cached_tokens or 0
            
            details = (
                getattr(response.usage, 'completion_tokens_details', None) or 
                getattr(response.usage, 'output_tokens_details', None)
            )
            if details and hasattr(details, 'reasoning_tokens'):
                usage.reasoning_tokens = details.reasoning_tokens or 0
        
        return usage

