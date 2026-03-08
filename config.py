from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    apify_enabled: bool = Field(default=False, alias="APIFY_ENABLED")
    apify_api_token: str = Field(default="", alias="APIFY_API_TOKEN")
    hunter_api_key: str = Field(default="", alias="HUNTER_API_KEY")
    snov_client_id: str = Field(default="", alias="SNOV_CLIENT_ID")
    snov_client_secret: str = Field(default="", alias="SNOV_CLIENT_SECRET")
    prospeo_api_key: str = Field(default="", alias="PROSPEO_API_KEY")
    gmail_client_id: str = Field(default="", alias="GMAIL_CLIENT_ID")
    gmail_client_secret: str = Field(default="", alias="GMAIL_CLIENT_SECRET")
    gmail_redirect_uri: str = Field(
        default="http://localhost:8000/api/auth/callback",
        alias="GMAIL_REDIRECT_URI",
    )
    database_url: str = Field(default="sqlite+aiosqlite:///./jobhunter.db", alias="DATABASE_URL")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    gmail_tokens_path: str = Field(default="./uploads/gmail_tokens.json", alias="GMAIL_TOKENS_PATH")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


settings = Settings()
