from pydantic_settings import BaseSettings
from pydantic import Field

class ApplicationSettings(BaseSettings):
    app_env: str = Field(default="local")
    api_rate_limit: int = Field(default=60)
    cache_ttl_seconds: int = Field(default=300)
    redis_url: str | None = Field(default=None)
    enable_audio_features: bool = Field(default=False)
    model_dir: str = Field(default=".models")

    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    s3_bucket: str | None = None

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

settings = ApplicationSettings()
