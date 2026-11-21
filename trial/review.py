"""本轮建议、修订记录与复盘快照。

* 每次运行写一份带 ``as_of``、``fingerprint`` 和 ``revision`` 的不可变快照；
  旧快照不覆盖，便于把“先前向门店引用过的结论”与“本轮修订”分开。
* ``--quote`` 把当前快照标记为‘已向门店引用’，之后的新快照不会改写它。
* 快照包含证据层、目标覆盖、决策、场次与待办、外推边界和资料质量。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .booking import gap_backlog, plan, suggestions
from .engine import Config, Selection, build_analysis, load


def _fingerprint(data: dict) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def merge(base: dict, extra: dict) -> dict:
    """把后到周报/退款流水按来源表追加，不改动原文件。"""
    out = {k: list(v) for k, v in base.items()}
    for table, rows in extra.items():
        if table not in out:
            out[table] = []
        known = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in out[table]}
        for row in rows:
            sig = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if sig not in known:
                out[table].append(row)
                known.add(sig)
    return out


def _quality(data: dict, analysis: dict) -> list[dict]:
    out = []
    a = analysis["audit"]
    if a["visits_unobserved"]:
        out.append(
            {
                "level": "warn",
                "code": "incomplete_windows",
                "message": f"{a['visits_unobserved']} 条到访记录窗口不完整，不进入分母。",
            }
        )
    if a["anonymous_observed"]:
        out.append(
            {
                "level": "info",
                "code": "anonymous_gap",
                "message": (
                    f"{a['anonymous_observed']} 条匿名曝光无连接键，"
                    "单列覆盖缺口，不推测成交。"
                ),
            }
        )
    if a["orders_unmatched"]:
        out.append(
            {
                "level": "info",
                "code": "unlinked_orders",
                "message": f"{len(a['orders_unmatched'])} 笔订单无法连到符合资格的曝光，未计入结果。",
            }
        )
    if a["orders_pending_refund"]:
        out.append(
            {
                "level": "warn",
                "code": "pending_refund",
                "message": (
                    f"{len(a['orders_pending_refund'])} 笔订单未到退款观察期，"
                    "未计净留单，待后续周报复核。"
                ),
            }
        )
    for a in analysis["assessments"]:
        if not a["target_mix"]:
            out.append(
                {
                    "level": "warn",
                    "code": "no_target_mix",
                    "message": f"{a['shop']} 没有目标客群构成，无法判断覆盖，先向合作方索取。",
                }
            )
    for c in analysis["cohorts"].values():
        if c["zero_conversion"]:
            out.append(
                {
                    "level": "info",
                    "code": "complete_zero",
                    "message": (
                        f"{c['label']} 完整计数 {c['mature']} 次、零成交；"
                        "这是结果，不是资料不足。"
                    ),
                }
            )
    return out


def build_report(
    data: dict,
    cfg: Config | None = None,
    selection: Selection | None = None,
    prior: str | None = None,
) -> dict:
    cfg = cfg or Config.from_intake(data["intake"])
    selection = selection or Selection()
    analysis = build_analysis(data, cfg, selection)

    gaps = []
    sessions = plan(data, cfg.as_of)
    for a in analysis["assessments"]:
        for sku, cov in a["coverage"].items():
            for g in cov["gaps"]:
                gaps.append(
                    {"shop": a["shop"], "sku": sku, "band": a["band"], **g}
                )
    # 同一层只保留最大目标权重
    unique: dict[tuple, dict] = {}
    for g in gaps:
        key = (g["shop"], g["band"], g["segment"], g["slot"])
        if key not in unique or g["mass"] > unique[key]["mass"]:
            unique[key] = g
    gaps = list(unique.values())

    return {
        "fingerprint": _fingerprint(data),
        "as_of": cfg.as_of,
        "selection": {
            "skus": list(selection.skus) if selection.skus
            else sorted({r["sku"] for r in data["listings"]}),
            "band": selection.band,
            "shops": list(selection.shops) if selection.shops
            else sorted({r["shop"] for r in data["target"]}),
        },
        "rules": {
            "purchase_days": cfg.purchase_days,
            "refund_days": cfg.refund_days,
            "min_mature": cfg.min_mature,
            "max_anon_share": cfg.max_anon_share,
            "max_pending_share": cfg.max_pending_share,
        },
        "prior_reference": prior,
        "cohorts": analysis["cohorts"],
        "assessments": analysis["assessments"],
        "sessions": [p.as_dict() for p in sessions],
        "suggestions": suggestions(sessions, gaps),
        "gap_backlog": gap_backlog(sessions, gaps),
        "audit": analysis["audit"],
        "quality": _quality(data, analysis),
        "cannot_extend": [
            "馆内到访结构不等合作方目标构成（target.mass 是客群权重，不是到访量）。",
            "匿名曝光只计覆盖缺口，成交率不可推测。",
            "covered_mass 之外的层不外推到整店。",
            "跨店率差过大或排序反向时，合并率不代表任一门店。",
            "已确认但未实际改订的承诺不随预测移动。",
            "未到退款观察期的订单不是净留单。",
        ],
    }


def write_snapshot(report: dict, out_dir: str | Path = "reports") -> Path:
    """写入不可变快照：同名文件已存在时追加修订号，绝不覆盖。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{report['as_of']}-{report['fingerprint']}"
    rev = 1
    while True:
        path = out / (f"{base}-r{rev}.json" if rev > 1 else f"{base}.json")
        if not path.exists():
            payload = dict(report, revision=path.name, quoted=False)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return path
        rev += 1


def quote(path: str | Path, note: str) -> Path:
    """把某个快照标记为已向门店引用；内容不变，只加引用记录。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["quoted"] = True
    data["quoted_note"] = note
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def revisions(out_dir: str | Path = "reports") -> list[dict]:
    """按时间顺序返回历史快照摘要，用于区分先前引用与本轮修订。"""
    rows = []
    # 快照按写入顺序排：同一输入重跑的修订号排在其后
    for path in sorted(Path(out_dir).glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name)):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "file": path.name,
                "as_of": r.get("as_of"),
                "fingerprint": r.get("fingerprint"),
                "quoted": r.get("quoted", False),
                "quoted_note": r.get("quoted_note"),
                "selection": r.get("selection"),
                "sessions": {s["window"]: s["status"] for s in r.get("sessions", [])},
                "open_actions": sorted(
                    {
                        action
                        for s in r.get("sessions", [])
                        for action in s.get("actions", [])
                    }
                ),
            }
        )
    return rows


def diff_revisions(old: dict, new: dict) -> dict:
    """比较两份快照，只列变化，不复述旧结论为当前结论。"""
    def flatten(report: dict) -> dict:
        out = {}
        for a in report.get("assessments", []):
            for d in a.get("decisions", []):
                out[f"{a['shop']}|{d['band']}|{d['sku']}"] = {
                    "category": d["category"],
                    "worth_discussing": d["worth_discussing"],
                    "whole_store_claim_allowed": d["whole_store_claim_allowed"],
                    "weighted_rate": d["weighted_rate"],
                }
        return out

    o, n = flatten(old), flatten(new)
    changes = []
    for key in sorted(set(o) | set(n)):
        if o.get(key) != n.get(key):
            changes.append({"key": key, "before": o.get(key), "after": n.get(key)})
    return {
        "from": old.get("file", old.get("as_of")),
        "to": new.get("file", new.get("as_of")),
        "selection_changed": old.get("selection") != new.get("selection"),
        "changes": changes,
        "old_was_quoted": old.get("quoted", False),
    }
