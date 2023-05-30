from pydantic import BaseSettings, Field

class STDBaseConfig(BaseSettings):
    """Define any config here.

    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """

    num_replicas: int = 1
    num_cpus: int = 4
    num_gpus: float = 0.0

    target_rate: int = 16000
    target_channels: int = 1

std_config = STDBaseConfig()