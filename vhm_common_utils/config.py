from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Kafka Settings
    kafka_bootstrap_servers: str = "localhost:9092"

    # Qdrant Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "anchors"

    # Embedding Settings
    embedding_model: str = "deterministic"
    ollama_base_url: str = "http://localhost:11434"
    portkey_api_key: str | None = None
    portkey_base_url: str = "https://api.portkey.ai/v1"

# Instantiate a single settings object for the application
settings = Settings()
