import json
import re
from types import SimpleNamespace
from pydantic import BaseModel

# Import your schemas (Statements, Verdicts, Reason, etc.)
# For example:
#
# from your_schema_module import Statements, Verdicts, Reason

# A helper to fix up the dictionary for required keys.
def fixup_required_keys(data: dict, schema: BaseModel):
    # Determine what key our schema requires.
    # You can adjust these as needed based on your actual Pydantic models.
    if schema.__name__ == "Statements":
        # Expected key: "statements", which should be a nonempty list
        if "statements" not in data or not data["statements"]:
            data["statements"] = ["idk"]
    elif schema.__name__ == "Verdicts":
        # Expected key: "verdicts", which should be a nonempty list
        if "verdicts" not in data or not data["verdicts"]:
            data["verdicts"] = [{"verdict": "idk", "reason": None}]
    elif schema.__name__ == "Reason":
        # Expected key: "reason"
        if "reason" not in data or not data["reason"]:
            data["reason"] = "idk"
    return data

def extract_required_json(text: str, expected_keys: list) -> dict:
    """
    Attempt to extract a JSON object that contains one of the expected keys.
    This function uses a simple approach: it looks for the first '{' that, when parsed,
    results in a dict containing one of the expected keys.
    """
    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)
    for candidate in json_objects:
        try:
            parsed = json.loads(candidate)
            if any(key in parsed for key in expected_keys):
                return parsed
        except Exception:
            continue
    # Fallback: return empty dict
    return {}

def parse_response(response_content, schema=None, debug=True):
    """
    Parse the LLM response content, printing debugging info.
    If a schema is provided, first try to instantiate it.
    On failure or if the expected key is missing, attempt to extract the proper JSON
    based only on expected keys. In case the required key is not present or empty,
    populate it with default value(s) ("idk").
    If no schema is provided, returns a SimpleNamespace wrapping the parsed dict.
    """
    if debug:
        print("DEBUG: Raw response content:")
        print(response_content)
    try:
        parsed = json.loads(response_content)
        if debug:
            print("DEBUG: Parsed JSON:")
            print(parsed)
    except Exception as e:
        if debug:
            print(f"DEBUG: Initial JSON parsing failed: {e}")
        # Fall back to extracting a JSON object using regex
        parsed = extract_required_json(response_content, expected_keys=["statements", "verdicts", "reason"])
        if debug:
            print("DEBUG: Extracted JSON via fallback:")
            print(parsed)
            
    if schema is not None:
        try:
            # Fix up parsed data for required keys.
            parsed = fixup_required_keys(parsed, schema)
            return schema(**parsed)
        except Exception as e:
            if debug:
                print("DEBUG: Schema validation error:", e)
                print("DEBUG: Attempting to extract required JSON from response content...")
            # Fallback: extract a JSON object that contains the expected key 
            expected = []
            if schema.__name__ == "Statements":
                expected = ["statements"]
            elif schema.__name__ == "Verdicts":
                expected = ["verdicts"]
            elif schema.__name__ == "Reason":
                expected = ["reason"]
            trimmed_data = extract_required_json(response_content, expected)
            if debug:
                print("DEBUG: Extracted trimmed JSON:")
                print(trimmed_data)
            trimmed_data = fixup_required_keys(trimmed_data, schema)
            return schema(**trimmed_data)
    else:
        return SimpleNamespace(**parsed)
