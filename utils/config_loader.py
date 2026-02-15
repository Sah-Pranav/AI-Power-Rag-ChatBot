import os
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage configuration"""
    
    _instance = None
    
    def __new__(cls, config_path: str = "config/config.yaml"):
        """Singleton pattern - only one config instance"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance.config_path = config_path
            cls._instance.config = cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config and substitute environment variables"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        return config
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute ${VAR} with environment variables"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config
    
    def get(self, *keys, default=None):
        """Get nested config value: config.get('embeddings', 'provider')"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()

# Create global config instance
def get_config():
    """Get the global config instance"""
    return ConfigLoader()

# For convenience, create a default instance
config = get_config()