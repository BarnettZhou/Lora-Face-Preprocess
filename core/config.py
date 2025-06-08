import json
import os
from typing import Dict, List, Any, Optional

class ConfigError(Exception):
    """配置错误异常"""
    pass

class Config:
    """配置管理类"""
    
    def __init__(self):
        self.default_config_path = "e:/codes/face-preprocess/config/default.json"
        self.custom_config_path = "e:/codes/face-preprocess/config/custome.json"
        
        # 配置约束定义
        self.valid_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.valid_formats = ["jpg", "png"]
        self.valid_types = ["face", "upper_body", "half_body"]
        self.valid_blanks = ["keep-blank", "fill-blank"]
        self.valid_sizes = {
            "1:1": [(512, 512), (768, 768), (1024, 1024)],
            "3:4": [(576, 768), (768, 1024)],
            "4:3": [(768, 576), (1024, 768)]
        }
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        try:
            with open(self.default_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"默认配置文件不存在: {self.default_config_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"默认配置文件格式错误: {e}")
    
    def _load_custom_configs(self) -> Dict[str, Any]:
        """加载自定义配置"""
        if not os.path.exists(self.custom_config_path):
            return {}
        try:
            with open(self.custom_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"自定义配置文件格式错误: {e}")
    
    def _save_custom_configs(self, configs: Dict[str, Any]):
        """保存自定义配置"""
        os.makedirs(os.path.dirname(self.custom_config_path), exist_ok=True)
        try:
            with open(self.custom_config_path, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise ConfigError(f"保存配置文件失败: {e}")
    
    def _validate_config(self, config: Dict[str, Any]):
        """验证配置的合法性"""
        # 检查必需字段
        required_fields = ["src_dir", "output_dir", "threshold", "size", "format", "types", "center", "blank"]
        for field in required_fields:
            if field not in config:
                raise ConfigError(f"缺少必需的配置项: {field}")
        
        # 验证threshold
        if config["threshold"] not in self.valid_thresholds:
            raise ConfigError(f"threshold必须是以下值之一: {self.valid_thresholds}")
        
        # 验证format
        if config["format"] not in self.valid_formats:
            raise ConfigError(f"format必须是以下值之一: {self.valid_formats}")
        
        # 验证types
        if not isinstance(config["types"], list) or not config["types"]:
            raise ConfigError("types必须是非空列表")
        for type_item in config["types"]:
            if type_item not in self.valid_types:
                raise ConfigError(f"types中的值必须是以下之一: {self.valid_types}")
        
        # 验证center
        if not isinstance(config["center"], bool):
            raise ConfigError("center必须是布尔值")
        
        # 验证blank
        if config["blank"] not in self.valid_blanks:
            raise ConfigError(f"blank必须是以下值之一: {self.valid_blanks}")
        
        # 验证size
        if not isinstance(config["size"], dict) or "width" not in config["size"] or "height" not in config["size"]:
            raise ConfigError("size必须包含width和height字段")
        
        width = config["size"]["width"]
        height = config["size"]["height"]
        
        # 检查尺寸是否在有效范围内
        size_valid = False
        for ratio, sizes in self.valid_sizes.items():
            if (width, height) in sizes:
                size_valid = True
                break
        
        if not size_valid:
            valid_sizes_str = ", ".join([f"{w}x{h}" for sizes in self.valid_sizes.values() for w, h in sizes])
            raise ConfigError(f"size必须是以下尺寸之一: {valid_sizes_str}")
    
    def create_config(self, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成配置对象
        
        Args:
            custom_config: 自定义配置项，会覆盖默认配置
            
        Returns:
            配置对象
            
        Raises:
            ConfigError: 配置不合法时抛出异常
        """
        # 加载默认配置
        config = self._load_default_config().copy()
        
        # 应用自定义配置
        if custom_config:
            config.update(custom_config)
        
        # 验证配置
        self._validate_config(config)
        
        return config
    
    def save_user_config(self, name: str, config: Dict[str, Any]):
        """保存用户配置
        
        Args:
            name: 配置名称
            config: 配置内容
            
        Raises:
            ConfigError: 配置不合法或保存失败时抛出异常
        """
        if not name or not name.strip():
            raise ConfigError("配置名称不能为空")
        
        # 验证配置
        self._validate_config(config)
        
        # 加载现有的自定义配置
        custom_configs = self._load_custom_configs()
        
        # 添加新配置
        custom_configs[name] = config
        
        # 保存配置
        self._save_custom_configs(custom_configs)
    
    def delete_config(self, name: str):
        """删除配置
        
        Args:
            name: 配置名称
            
        Raises:
            ConfigError: 配置不存在或删除失败时抛出异常
        """
        if not name or not name.strip():
            raise ConfigError("配置名称不能为空")
        
        # 加载现有的自定义配置
        custom_configs = self._load_custom_configs()
        
        if name not in custom_configs:
            raise ConfigError(f"配置 '{name}' 不存在")
        
        # 删除配置
        del custom_configs[name]
        
        # 保存配置
        self._save_custom_configs(custom_configs)
    
    def load_config(self, name: Optional[str] = None) -> Dict[str, Any]:
        """读取配置
        
        Args:
            name: 配置名称，为空时返回默认配置
            
        Returns:
            配置对象
            
        Raises:
            ConfigError: 配置不存在或加载失败时抛出异常
        """
        if not name or not name.strip():
            # 返回默认配置
            return self.create_config()
        
        # 加载自定义配置
        custom_configs = self._load_custom_configs()
        
        if name not in custom_configs:
            raise ConfigError(f"配置 '{name}' 不存在")
        
        config = custom_configs[name]
        
        # 验证配置（防止文件被手动修改导致配置不合法）
        self._validate_config(config)
        
        return config
    
    def list_configs(self) -> List[str]:
        """列出所有自定义配置名称
        
        Returns:
            配置名称列表
        """
        custom_configs = self._load_custom_configs()
        return list(custom_configs.keys())