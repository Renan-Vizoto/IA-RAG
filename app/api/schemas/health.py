from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"ok": "ok"}]})

    ok: str = Field(description="Static liveness marker.")
