"""Debug Prospeo search."""
import asyncio
import httpx
import json
from config import settings

async def test():
    headers = {
        "Authorization": f"Bearer {settings.prospeo_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        resp = await client.post(
            "https://api.prospeo.io/search-person",
            json={
                "page": 1,
                "per_page": 25,
                "filters": {
                    "company": {
                        "websites": {
                            "include": ["google.com"],
                        }
                    }
                },
            },
        )
        print(f"Status: {resp.status_code}")
        print(f"Body: {json.dumps(resp.json(), indent=2)}")

asyncio.run(test())
