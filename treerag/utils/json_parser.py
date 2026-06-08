import json


def safe_parse_json(text: str) -> dict | list:
    # try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # sometimes the model wraps JSON in markdown code fences, strip them
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # drop the opening fence line and closing fence
        inner = "\n".join(lines[1:])
        inner = inner.rsplit("```", 1)[0].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # last resort: find the first { or [ and the matching closing bracket
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON from model response:\n{text[:300]}")
