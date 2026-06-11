from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

try:
    from scalar_fastapi import (
        AgentScalarConfig,
        DocumentDownloadType,
        Layout,
        SearchHotKey,
        get_scalar_api_reference,
    )
except Exception:  # pragma: no cover - dependency is installed in the API image.
    AgentScalarConfig = None
    DocumentDownloadType = None
    Layout = None
    SearchHotKey = None
    get_scalar_api_reference = None


API_DESCRIPTION = """
Consumer reference for the Dutch Energy RAG API.

The API exposes agentic RAG chat over Dutch residential energy pipeline knowledge.

Use `POST /chat/message` for conversational experiences, `POST /chat/chats` and
`GET /chat/chats` to manage chat threads, and `GET /chat/chats/{chat_id}/messages`
for history.

Session behavior: chat uses an `HttpOnly` `session_id` cookie for anonymous
visitors. If the client does not send one, the API creates it and returns
`Set-Cookie`. Each session can have multiple chats; pass `chat_id` in
`POST /chat/message` or rely on the `chat_id` HttpOnly cookie. When no chat
exists yet, the API creates one and returns it in the response body and cookie.
Token usage is tracked per turn (`tokens`) and cumulatively per chat (`chat_tokens`).
"""

OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Operational liveness checks for load balancers and local smoke tests.",
    },
    {
        "name": "chat",
        "description": (
            "Agentic RAG conversation endpoints. Chat can invoke semantic search, "
            "returns a `response_id`, and tracks token usage."
        ),
    },
]

SCALAR_CUSTOM_CSS = """
:root {
  --scalar-radius: 8px;
}

.dark-mode {
  --scalar-color-accent: #1f9d8a;
  --scalar-color-accent-hover: #2fb6a0;
}
"""


def register_docs_routes(app: FastAPI) -> None:
    @app.get("/docs", include_in_schema=False)
    async def scalar_docs():
        if get_scalar_api_reference is not None:
            return get_scalar_api_reference(
                openapi_url=app.openapi_url,
                title=f"{app.title} Reference",
                layout=Layout.MODERN,
                dark_mode=True,
                hide_dark_mode_toggle=False,
                show_sidebar=True,
                hide_search=False,
                hide_models=False,
                hide_test_request_button=False,
                show_developer_tools="always",
                document_download_type=DocumentDownloadType.BOTH,
                default_open_all_tags=True,
                expand_all_responses=True,
                expand_all_model_sections=True,
                order_schema_properties_by="preserve",
                search_hot_key=SearchHotKey.K,
                telemetry=False,
                custom_css=SCALAR_CUSTOM_CSS,
                servers=[
                    {
                        "url": "http://localhost:3333",
                        "description": "Local Docker Compose",
                    }
                ],
                agent=AgentScalarConfig(disabled=True),
            )

        return HTMLResponse(
            """
            <!doctype html>
            <html>
              <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Dutch Energy RAG API Reference</title>
                <style>body{margin:0}</style>
              </head>
              <body>
                <div id="app"></div>
                <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
                <script>
                  Scalar.createApiReference("#app", {
                    url: "/openapi.json",
                    layout: "modern",
                    darkMode: true,
                    showSidebar: true,
                    hideSearch: false,
                    hideModels: false,
                    hideTestRequestButton: false,
                    defaultOpenAllTags: true,
                    expandAllResponses: true,
                    expandAllModelSections: true,
                    orderSchemaPropertiesBy: "preserve",
                    telemetry: false,
                    servers: [
                      {
                        url: "http://localhost:3333",
                        description: "Local Docker Compose"
                      }
                    ]
                  })
                </script>
              </body>
            </html>
            """,
        )

    @app.get("/swagger", include_in_schema=False)
    async def swagger_docs():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} Swagger UI",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_docs():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} ReDoc",
        )


def configure_openapi(app: FastAPI) -> None:
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            summary="RAG chat for Dutch energy pipeline analysis.",
            description=API_DESCRIPTION,
            routes=app.routes,
            tags=OPENAPI_TAGS,
        )
        schema["servers"] = [
            {
                "url": "http://localhost:3333",
                "description": "Local Docker Compose",
            }
        ]
        schema["externalDocs"] = {
            "description": "OpenAPI document",
            "url": "/openapi.json",
        }
        schema["x-tagGroups"] = [
            {
                "name": "Core API",
                "tags": ["chat"],
            },
            {
                "name": "Operations",
                "tags": ["health"],
            },
        ]

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def build_error_responses(
    *codes: int,
    extra: dict[int, dict[str, Any]] | None = None,
) -> dict[int | str, dict[str, Any]]:
    catalog: dict[int, dict[str, Any]] = {
        400: {
            "description": "Invalid client request.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Modelo 'foo' nao permitido. Modelos disponiveis: gemma4-unsloth"
                    }
                }
            },
        },
        404: {
            "description": "Resource not found.",
            "content": {
                "application/json": {
                    "example": {"detail": "Resposta nao encontrada"}
                }
            },
        },
        422: {
            "description": "Request validation error generated by FastAPI.",
        },
        500: {
            "description": "Unhandled infrastructure, model, or persistence failure.",
        },
        503: {
            "description": "A required API service was not initialized.",
            "content": {
                "application/json": {
                    "example": {"detail": "Chat nao inicializado"}
                }
            },
        },
    }
    responses = {code: catalog[code] for code in codes if code in catalog}
    if extra:
        responses.update(extra)
    return responses
