import asyncio


async def main() -> None:
    print("1. Testing Arbeitnow API (no key)...")
    from tools.arbeitnow_tool import search_arbeitnow

    jobs = await search_arbeitnow("python developer")
    print(f"   Found {len(jobs)} jobs")

    print("2. Testing RemoteOK API (no key)...")
    from tools.remoteok_tool import search_remoteok

    jobs = await search_remoteok("python")
    print(f"   Found {len(jobs)} jobs")

    print("3. Testing Apify LinkedIn (free plan)...")
    try:
        from tools.apify_tool import search_linkedin_jobs

        jobs = await search_linkedin_jobs("software engineer", "Remote", 3)
        print(f"   Found {len(jobs)} jobs")
    except Exception as exc:
        print(f"   Skipped Apify test: {exc}")

    print("4. Testing Email Finder (Snov -> Prospeo -> patterns)...")
    from tools.email_finder_tool import find_hr_email

    result = await find_hr_email("Google", "google.com")
    print(f"   Result: {result}")

    print("All tools tested successfully!")


if __name__ == "__main__":
    asyncio.run(main())
