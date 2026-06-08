import os
import time
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 10


def call_llm(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system if system else None,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < MAX_RETRIES:
                    print(f"  Rate limit hit, waiting {RETRY_WAIT_SECONDS}s before retry {attempt}/{MAX_RETRIES - 1}...")
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    raise RuntimeError("Rate limit exceeded after all retries.") from e
            else:
                raise