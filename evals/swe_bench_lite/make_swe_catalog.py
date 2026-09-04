import pandas as pd

src = r"evals\swe_bench_lite\raw\dev.parquet"
dst = r"evals\swe_bench_lite\task_catalog.csv"

df = pd.read_parquet(src)

def count_patch_files(patch):
    patch = patch or ""
    return sum(
        line.startswith("diff --git ")
        for line in patch.splitlines()
    )

def count_patch_lines(patch):
    patch = patch or ""
    count = 0

    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            count += 1
        elif line.startswith("-") and not line.startswith("---"):
            count += 1

    return count

df["patch_files"] = df["patch"].map(count_patch_files)
df["patch_lines"] = df["patch"].map(count_patch_lines)
df["FTP"] = df["FAIL_TO_PASS"].map(len)
df["PTP"] = df["PASS_TO_PASS"].map(len)

out = df[
    [
        "instance_id",
        "repo",
        "difficulty",
        "patch_files",
        "patch_lines",
        "FTP",
        "PTP",
        "image",
    ]
]

out.to_csv(dst, index=False, encoding="utf-8-sig")

print(out.to_string(index=False))
print()
print(f"Saved: {dst}")
print(f"TOTAL: {len(out)}")
