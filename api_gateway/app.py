import os

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="Watinakama API Gateway")

BACKENDS = {
    "vin": {
        "base_url": os.environ.get("VIN_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        "token": None,
    },
    "body": {
        "base_url": os.environ.get("BODY_API_URL", "http://127.0.0.1:8080").rstrip("/"),
        "token": None,
    },
    "engine": {
        "base_url": os.environ.get("ENGINE_API_URL", "http://127.0.0.1:5003").rstrip("/"),
        "token": os.environ.get("ENGINE_API_TOKEN"),
    },
    "valuation": {
        "base_url": os.environ.get("VALUATION_API_URL", "http://127.0.0.1:5004").rstrip("/"),
        "token": os.environ.get("VALUATION_API_TOKEN"),
    },
}

# Headers that must not be forwarded verbatim between hops (RFC 7230) plus
# ones we recompute ourselves (host/content-length change once re-sent).
REQUEST_HEADER_EXCLUDE = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "authorization",
}
# httpx already decodes any compressed backend response, so the original
# content-encoding/content-length no longer describe `backend_response.content`.
RESPONSE_HEADER_EXCLUDE = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-encoding", "content-length",
}

client = httpx.AsyncClient(timeout=120.0)


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


@app.get("/health")
async def health():
    return {"status": "ok"}


async def proxy(service: str, path: str, request: Request) -> Response:
    backend = BACKENDS[service]
    target_url = f"{backend['base_url']}/{path}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in REQUEST_HEADER_EXCLUDE
    }
    if backend["token"]:
        headers["Authorization"] = f"Bearer {backend['token']}"

    body = await request.body()

    try:
        backend_response = await client.request(
            request.method,
            target_url,
            params=request.query_params,
            headers=headers,
            content=body,
        )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "message": f"Upstream {service} service unreachable: {exc}"},
        )

    response_headers = {
        k: v for k, v in backend_response.headers.items()
        if k.lower() not in RESPONSE_HEADER_EXCLUDE
    }
    return Response(
        content=backend_response.content,
        status_code=backend_response.status_code,
        headers=response_headers,
        media_type=backend_response.headers.get("content-type"),
    )


@app.api_route("/vin/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def vin_proxy(path: str, request: Request):
    return await proxy("vin", path, request)


@app.api_route("/body/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def body_proxy(path: str, request: Request):
    return await proxy("body", path, request)


@app.api_route("/engine/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def engine_proxy(path: str, request: Request):
    return await proxy("engine", path, request)


@app.api_route("/valuation/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def valuation_proxy(path: str, request: Request):
    return await proxy("valuation", path, request)
