import asyncio


async def main():
    print("=== Testing Fallback Job Sources ===\n")

    print("1. Testing Arbeitnow (free, no key)...")
    from tools.arbeitnow_tool import search_arbeitnow
    a_jobs = await search_arbeitnow("software engineer")
    print(f"   Arbeitnow found: {len(a_jobs)} jobs")
    for j in a_jobs[:3]:
        print(f"   - {j['title']} at {j['company']}")

    print("\n2. Testing RemoteOK (free, no key)...")
    from tools.remoteok_tool import search_remoteok
    r_jobs = await search_remoteok("software engineer")
    print(f"   RemoteOK found: {len(r_jobs)} jobs")
    for j in r_jobs[:3]:
        print(f"   - {j['title']} at {j['company']}")

    print("\n3. Testing LinkedIn search (Apify + guest)...")
    from tools.apify_tool import search_linkedin_jobs, get_linkedin_provider_reason
    l_jobs = await search_linkedin_jobs("software engineer", "Remote", 3)
    print(f"   LinkedIn found: {len(l_jobs)} jobs")
    print(f"   Provider reason: {get_linkedin_provider_reason()}")
    for j in l_jobs[:3]:
        print(f"   - {j['title']} at {j['company']} [{j['source']}]")

    print("\n4. Checking Gmail tokens...")
    from pathlib import Path
    token_path = Path("uploads/gmail_tokens.json")
    if token_path.exists():
        import json
        tokens = json.loads(token_path.read_text())
        has_token = bool(tokens.get("token"))
        has_refresh = bool(tokens.get("refresh_token"))
        print(f"   Token exists: {has_token}")
        print(f"   Refresh token exists: {has_refresh}")
    else:
        print("   gmail_tokens.json NOT FOUND - need to connect Gmail first!")

    total = len(a_jobs) + len(r_jobs) + len(l_jobs)
    print(f"\n=== TOTAL JOBS FROM ALL SOURCES: {total} ===")
    if total > 0:
        print("SUCCESS: Job search pipeline has working sources!")
    else:
        print("WARNING: No sources returned jobs!")


if __name__ == "__main__":
    asyncio.run(main())
