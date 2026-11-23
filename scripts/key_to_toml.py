"""
Prints a service-account JSON key as the [gcp_service_account] block for Streamlit secrets.

Usage:  python scripts/key_to_toml.py path/to/key.json
The output contains the private key: paste it into the Streamlit Cloud secrets box, do not save it in the repo.
"""

import json
import sys

if len(sys.argv) != 2:
    sys.exit("usage: python scripts/key_to_toml.py path/to/key.json")
with open(sys.argv[1], encoding="utf-8") as f:
    key = json.load(f)
if key.get("type") != "service_account":
    sys.exit("this is not a service-account key (field 'type' must be 'service_account')")
print("[gcp_service_account]")
for name, value in key.items():
    print(f"{name} = {json.dumps(value)}")  # JSON string escaping is valid TOML basic-string escaping
