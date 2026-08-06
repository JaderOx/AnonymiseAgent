import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

_CONFIGS_DIR = Path(__file__).parent


# ── 原有：ASR 模型参数配置（asr_model_config.yaml） ───────────────────────────

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.configs = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        if self.config_path.suffix in ['.yaml', '.yml']:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif self.config_path.suffix == '.json':
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {self.config_path.suffix}")

    def get_config(self, model_name: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        models = self.configs.get('models', {})
        if model_name not in models:
            raise ValueError(f"未知的模型名称: {model_name}")
        return models[model_name].copy()

    def update_config(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """更新配置（运行时动态修改）"""
        config = self.get_config(model_name)
        config.update(kwargs)
        return config


# ── 新增：语言 → ASR 模型映射（language_asr_map.yaml） ───────────────────────

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# 语言名标准化：英文/ISO → 中文（与 YAML 配置中的 key 对齐）
_LANG_ALIASES = {
    "chinese": "中文", "zh": "中文",
    "english": "英文", "en": "英文",
    "japanese": "日文", "ja": "日文",
    "spanish": "西班牙语", "es": "西班牙语",
    "german": "德语", "de": "德语",
    "french": "法语", "fr": "法语",
    "russian": "俄语", "ru": "俄语",
    "korean": "韩语", "ko": "韩语",
    "portuguese": "葡萄牙语", "pt": "葡萄牙语",
    "turkish": "土耳其语", "tr": "土耳其语",
    "dutch": "荷兰语", "nl": "荷兰语",
    "arabic": "阿拉伯语", "ar": "阿拉伯语",
    "swedish": "瑞典语", "sv": "瑞典语",
    "italian": "意大利语", "it": "意大利语",
    "polish": "波兰语", "pl": "波兰语",
}


def _normalize_language(language: str) -> str:
    """将各种语言名标准化为 YAML 配置中使用的中文 key。"""
    if not language:
        return language
    # 先查别名表
    normed = _LANG_ALIASES.get(language.lower().strip())
    if normed:
        return normed
    # 已经是中文，直接返回
    return language


_asr_map_cache: Optional[Dict] = None

def get_asr_config(language: str) -> Dict[str, Any]:
    """
    根据语言返回 ASR 配置字典。

    Returns:
        {
            'model': str,                  # ASR 模型名称
            'supports_hotwords': bool,
            'supports_speaker': bool,
            'is_default': bool,            # True 表示未在 languages 中找到，使用默认值
        }
    """
    global _asr_map_cache
    if _asr_map_cache is None:
        _asr_map_cache = _load_yaml(_CONFIGS_DIR / 'language_asr_map.yaml')

    language = _normalize_language(language)
    languages = _asr_map_cache.get('languages', {})
    if language in languages:
        cfg = dict(languages[language])
        cfg.setdefault('supports_hotwords', True)
        cfg.setdefault('supports_speaker', False)
        cfg['is_default'] = False
        return cfg

    # 未匹配到，使用 default
    default_model = _asr_map_cache.get('default', 'whisperx')
    return {
        'model': default_model,
        'supports_hotwords': True,
        'supports_speaker': False,
        'is_default': True,
    }


# ── 新增：语言 → LLM 模型 + prompt 映射（language_llm_map.yaml） ─────────────

_llm_map_cache: Optional[Dict] = None

def get_llm_config(language: str) -> Dict[str, Any]:
    """
    根据语言返回 LLM 配置字典。

    Returns:
        {
            'model': str,
            'prompt': str,
            'prompt_suffix': str,  # 已将 {language} 替换为实际语言名
            'group_prefix': str,
            'is_default': bool,
        }
    """
    global _llm_map_cache
    if _llm_map_cache is None:
        _llm_map_cache = _load_yaml(_CONFIGS_DIR / 'language_llm_map.yaml')

    language = _normalize_language(language)
    languages = _llm_map_cache.get('languages', {})
    default   = _llm_map_cache.get('default', {})

    raw = dict(languages.get(language) or default)
    is_default = language not in languages

    # 如果只有模型名（即 default 块只写了 model），从 default 块补全其他字段
    if isinstance(default, dict):
        for key in ('prompt', 'prompt_suffix', 'group_prefix'):
            raw.setdefault(key, default.get(key, ''))

    # 替换 prompt_suffix 中的 {language} 占位符
    suffix = raw.get('prompt_suffix', '')
    raw['prompt_suffix'] = suffix.replace('{language}', language)

    raw['is_default'] = is_default
    return raw
