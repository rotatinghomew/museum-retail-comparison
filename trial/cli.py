"""命令行入口。

示例
----
python3 -m trial.cli --data data/drill.json --skus 纸域-v1 陶语-v1 --band 59档 \
    --shops 合作南店 --write
python3 -m trial.cli --append data/week2.json --as-of 2026-07-30 --write
python3 -m trial.cli --history
python3 -m trial.cli --diff-last
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Config, Selection, load
from .review import (
    build_report,
    diff_revisions,
    merge,
    quote,
    revisions,
    write_snapshot,
)


def _load_history(files: list[Path]) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="馆内试销 -> 合作门店 可比证据复盘")
    ap.add_argument("--data", default="data/drill.json")
    ap.add_argument("--append", action="append", default=[], help="后到周报/退款流水 JSON")
    ap.add_argument("--as-of", default=None, help="覆盖 intake.as_of")
    ap.add_argument("--skus", nargs="*", default=[])
    ap.add_argument("--band", default=None)
    ap.add_argument("--shops", nargs="*", default=[])
    ap.add_argument("--min-mature", type=int, default=None)
    ap.add_argument("--max-anon-share", type=float, default=None)
    ap.add_argument("--max-pending-share", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--prior", default=None, help="本轮修订所基于的上一份快照文件名")
    ap.add_argument("--write", action="store_true", help="写入 reports/ 不可变快照")
    ap.add_argument("--quote", metavar="FILE", default=None,
                    help="把指定快照标记为已向门店引用")
    ap.add_argument("--quote-note", default="")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--diff-last", action="store_true")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    if args.quote:
        path = quote(args.quote, args.quote_note)
        print(f"已标记引用 {path}")
        return 0

    if args.history:
        print(json.dumps(revisions(args.out), ensure_ascii=False, indent=2))
        return 0

    if args.diff_last:
        files = sorted(Path(args.out).glob("*.json"))
        if len(files) < 2:
            print("快照不足两份，无法比较。", file=sys.stderr)
            return 2
        old, new = _load_history(files[-2:])
        print(json.dumps(diff_revisions(old, new), ensure_ascii=False, indent=2))
        return 0

    data = load(args.data)
    for extra in args.append:
        data = merge(data, load(extra))

    overrides = {}
    for name in ("as_of", "min_mature", "max_anon_share", "max_pending_share", "bootstrap"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    cfg = Config.from_intake(data["intake"], **overrides)
    selection = Selection(
        skus=tuple(args.skus), band=args.band, shops=tuple(args.shops)
    )

    report = build_report(data, cfg, selection, prior=args.prior)
    if args.write:
        path = write_snapshot(report, args.out)
        report["file"] = path.name
        print(f"写入 {path}", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
