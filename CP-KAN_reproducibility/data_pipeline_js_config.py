from dataclasses import dataclass
from typing import Optional, List

import yaml


@dataclass
class DataConfig:
    data_path: str
    n_rows:int
    train_ratio: float
    feature_cols: List[str]
    target_col: str
    weight_col: str
    date_col:str

    @classmethod
    def from_dict(cls, data: dict) -> 'DataConfig':
        return cls(**data)

@dataclass
class ModelConfig:
    model_type: str
    network_shape: List[int]
    max_degree: Optional[int] = None
    complexity_weight: Optional[float] = None
    significance_threshold: Optional[float] = None
    hidden_dims: Optional[List[int]] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    n_epochs: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelConfig':
        return cls(**data)
@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    models: List[ModelConfig]
    random_seed: int
    num_trials:int
    save_path: str
    log_path: str

    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentConfig':
        # Convert nested dicts to appropriate dataclass instances
        data_config = DataConfig.from_dict(data['data'])
        model_configs = [ModelConfig.from_dict(model_dict) for model_dict in data['models']]

        return cls(
            name=data['name'],
            data=data_config,
            models=model_configs,
            random_seed=data['random_seed'],
            num_trials=data['num_trials'],
            save_path=data['save_path'],
            log_path=data['log_path']
        )

def load_config(path: str) -> ExperimentConfig:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict['data']['feature_cols'] == 'auto':
        config_dict['data']['feature_cols'] = get_default_features()
    return ExperimentConfig.from_dict(config_dict)

def get_default_features():
    return [f'feature_{i:02d}' for i in range(79)]
