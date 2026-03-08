"""Debug Snov.io domain search step by step."""
import asyncio
import httpx
import json
from config import settings

async def test():
    async with httpx.AsyncClient(timeout=20) as client:
        # Auth
        resp = await client.post(
            "https://api.snov.io/v1/oauth/access_token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.snov_client_id,
                "client_secret": settings.snov_client_secret,
            },
        )
        token = resp.json().get("access_token")
        print(f"Token: {token[:20]}..." if token else "Token: NONE")

        # Start domain search
        resp2 = await client.get(
            "https://api.snov.io/v2/domain-search/domain-emails/start",
            params={"access_token": token, "domain": "google.com"},
        )
        print(f"\nStart Status: {resp2.status_code}")
        print(f"Start Body: {json.dumps(resp2.json(), indent=2)}")

asyncio.run(test())
