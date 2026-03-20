"""
SLURM script validation for detecting typos and syntax errors in SBATCH directives.
"""

import re
from difflib import get_close_matches

VALID_SBATCH_DIRECTIVES = {
    # Job identification
    "job-name", "J",
    # Output
    "output", "o", "error", "e",
    # Resources
    "nodes", "N", "ntasks", "n", "ntasks-per-node",
    "cpus-per-task", "c", "mem", "mem-per-cpu", "mem-per-gpu",
    "time", "t", "gres", "gpus", "gpus-per-node", "gpus-per-task",
    # Node selection
    "partition", "p", "nodelist", "w", "exclude", "x", "constraint", "C",
    # Account/QOS
    "account", "A", "qos", "q", "reservation",
    # Notifications
    "mail-user", "mail-type",
    # Arrays/dependencies
    "array", "a", "dependency", "d",
    # Other
    "exclusive", "export", "chdir", "D", "wait", "begin", "B",
    "comment", "open-mode", "test-only", "parsable", "verbose",
}


def validate_script(script: str) -> list[dict]:
    """Find typos and syntax errors in SBATCH directives."""
    errors = []

    for line in script.split('\n'):
        line = line.strip()
        if not line.startswith('#SBATCH'):
            continue

        # Extract directive and value
        match = re.match(r'#SBATCH\s+--?(\S+?)(?:=(.*))?$', line)
        if not match:
            continue

        directive = match.group(1).split('=')[0]
        value = match.group(2) or ""

        # Check directive name
        if directive not in VALID_SBATCH_DIRECTIVES:
            suggestion = get_close_matches(directive, VALID_SBATCH_DIRECTIVES, n=1)
            errors.append({
                "type": "invalid_directive",
                "found": directive,
                "suggestion": suggestion[0] if suggestion else None,
                "line": line
            })

        # Check mem value format (e.g., 8GG is invalid)
        if directive in ("mem", "mem-per-cpu", "mem-per-gpu") and value:
            if not re.match(r'^\d+[KMGT]?$', value, re.IGNORECASE):
                errors.append({
                    "type": "invalid_value",
                    "directive": directive,
                    "found": value,
                    "line": line
                })

    return errors


def format_errors_for_llm(errors: list[dict]) -> str:
    """Format validation errors for inclusion in LLM context."""
    if not errors:
        return ""

    lines = ["[SCRIPT VALIDATION - The following issues were detected. Quote these exact values in your response:]"]
    for e in errors:
        if e["type"] == "invalid_directive":
            if e["suggestion"]:
                lines.append(f"- Typo: '{e['found']}' should be '{e['suggestion']}'")
            else:
                lines.append(f"- '{e['found']}' is not a valid SBATCH directive.")
        elif e["type"] == "invalid_value":
            lines.append(f"- Invalid value: '{e['found']}' for --{e['directive']} (mention '{e['found']}' in your response)")

    return "\n".join(lines)
