from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "prithivMLmodi/AI-or-Not"
    device: str = "cpu"
    max_file_size_mb: int = 16
    rate_limit_per_minute: int = 20
    allowed_extensions: set[str] = {"jpg", "jpeg", "png", "webp"}
    verdict_ai_threshold: float = 0.7
    verdict_real_threshold: float = 0.3

    model_config = {"env_prefix": "JAP_"}


settings = Settings()
