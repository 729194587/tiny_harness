import json
from pathlib import Path


SOURCE = Path("evals/swe_bench_lite/dev.jsonl")
OUTPUT = Path("evals/swe_bench_lite/selected_tasks.jsonl")

SELECTED = {
    "marshmallow-code__marshmallow-1343",
    "pylint-dev__astroid-1196",
    "pydicom__pydicom-1139",
    "sqlfluff__sqlfluff-1517",
}


selected_rows = []

with SOURCE.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        if row["instance_id"] in SELECTED:
            selected_rows.append(row)


found = {row["instance_id"] for row in selected_rows}
missing = SELECTED - found

if missing:
    raise RuntimeError(f"Missing tasks: {sorted(missing)}")


with OUTPUT.open("w", encoding="utf-8") as f:
    for row in selected_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


print(f"Saved {len(selected_rows)} tasks to {OUTPUT}")

for row in selected_rows:
    print("-", row["instance_id"])