import asyncio
import json
from pathlib import Path
from tools.resume_parser import _llm_extract_resume

async def main():
    text = Path("cv_raw_text.txt").read_text(encoding="utf-8")
    print("Testing parser...")
    parsed = await _llm_extract_resume(text)
    
    clean = {k: v for k, v in parsed.items() if k != "raw_text"}
    print(json.dumps(clean, indent=2))

asyncio.run(main())
