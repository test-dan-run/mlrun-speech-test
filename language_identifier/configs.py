from pydantic import BaseSettings, Field

class LIDBaseConfig(BaseSettings):
    """Define any config here.

    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """

    num_replicas: int = 1
    num_cpus: int = 4
    num_gpus: float = 0.0

    lid_model_path: str = '/models'

lid_config = LIDBaseConfig()