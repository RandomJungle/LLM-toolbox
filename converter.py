import json
from typing import Dict


def convert_content_to_json(content: str) -> Dict:
    json_content = content.replace(
        '```json', ''
    ).replace(
        '```', ''
    )
    return json.loads(json_content)