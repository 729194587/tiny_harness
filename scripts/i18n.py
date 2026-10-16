"""Check whole-file bilingual review records; only `record` writes files."""

import argparse
import hashlib
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"README", "docs/runtime", "docs/context-attribution",
            "evals/swe_bench_lite/README"}
FIELDS = {"en_sha256", "zh_CN_sha256"}


def discover(root):
    """Include future reader docs, but never operational or historical files."""
    files = list(root.glob("*"))
    for directory in ("docs", "evals/swe_bench_lite"):
        files.extend((root / directory).rglob("*"))
    pairs = set(REQUIRED)
    for path in files:
        relative = path.relative_to(root)
        if relative.parts[:3] == ("docs", "superpowers", "plans"):
            continue
        if path.name in {"AGENTS.md", "SKILL.md", "AGENTS.zh-CN.md", "SKILL.zh-CN.md"}:
            continue
        for suffix in (".zh-CN.md", ".md", ".i18n.yaml"):
            if path.is_file() and path.name.endswith(suffix):
                pairs.add(relative.as_posix()[:-len(suffix)])
                break
    return sorted(pairs)


def pair_paths(root, pair):
    return [root / (pair + suffix) for suffix in (".md", ".zh-CN.md", ".i18n.yaml")]


def hashes(paths):
    return dict(zip(("en_sha256", "zh_CN_sha256"),
                    (hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)))


class RecordLoader(yaml.SafeLoader):
    """Reject duplicate fields instead of silently using the last hash."""


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("record keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


RecordLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def read_record(path):
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=RecordLoader)
    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError("expected only en_sha256 and zh_CN_sha256")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in data.values()):
        raise ValueError("hashes must be 64 lowercase hexadecimal characters")
    return data


def check(root):
    errors = []
    for pair in discover(root):
        en, zh, record = pair_paths(root, pair)
        missing = [p.name for p in (en, zh) if not p.is_file()]
        if missing:
            errors.append(f"{pair}: 缺少语言文件: {', '.join(missing)}")
            if record.exists():
                errors.append(f"{pair}: 无法识别记录（缺少完整文档对）")
        if not record.is_file():
            errors.append(f"{pair}: 缺少记录 {record.name}")
            continue
        try:
            saved = read_record(record)
            if missing:
                continue
            current = hashes((en, zh))
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
            errors.append(f"{pair}: 无效记录或无法读取文件: {error}")
            continue
        changed = [label for key, label in (("en_sha256", "English"),
                                            ("zh_CN_sha256", "简体中文"))
                   if saved[key] != current[key]]
        if changed:
            errors.append(f"{pair}: {'、'.join(changed)} 已变更；请共同审阅两种语言后执行 "
                          f"record {pair}，不会自动协调或更新记录")
    return errors


def record(root, pair):
    # Accept the pair stem or its English filename, always relative to the repo.
    if pair.endswith(".md"):
        pair = pair[:-3]
    if pair not in discover(root):
        raise ValueError(f"不在维护范围内的文档对: {pair}")
    en, zh, target = pair_paths(root, pair)
    if not en.is_file() or not zh.is_file():
        raise ValueError(f"{pair}: 两种语言文件必须都存在，才能记录审阅状态")
    data = hashes((en, zh))
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="只读检查所有文档对")
    update = commands.add_parser("record", help="确认两种语言已共同审阅，并记录当前哈希")
    update.add_argument("pair", help="仓库相对路径，例如 docs/runtime 或 docs/runtime.md")
    args = parser.parse_args(argv)
    if args.command == "check":
        errors = check(ROOT)
        for error in errors:
            print(error)
        if not errors:
            print(f"双语记录检查通过：{len(discover(ROOT))} 对文档")
        return int(bool(errors))
    try:
        pair = record(ROOT, args.pair)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"已记录共同审阅状态：{pair}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
