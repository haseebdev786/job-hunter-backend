"""Quick test of the rewritten email finder pipeline."""
import asyncio
import json
from tools.email_finder_tool import find_hr_email

async def main():
    companies = [
        ("Google", "google.com"),
        ("Systems Limited", "systemsltd.com"),
        ("10Pearls", "10pearls.com"),
        ("Devsinc", "devsinc.com"),
        ("Arbisoft", "arbisoft.com"),
    ]
    for company, domain in companies:
        print(f"\n{'='*60}")
        print(f"Company: {company} | Domain: {domain}")
        print(f"{'='*60}")
        result = await find_hr_email(company, domain)
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
