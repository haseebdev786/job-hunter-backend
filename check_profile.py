import asyncio
from models.database import async_session, Profile

async def main():
    async with async_session() as db:
        p = await db.get(Profile, 1)
        raw = p.data.get("raw_text", "")
        with open("cv_raw_text.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print(f"Wrote {len(raw)} chars to cv_raw_text.txt")
        print("First 1000 chars:")
        print(raw[:1000])

asyncio.run(main())
