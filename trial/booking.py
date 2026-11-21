"""场次、承诺与库存变更。

规则
----
* 预约按 ``window`` + ``sku`` 记占位；``availability`` 按 ``known_on`` 追加，
  取分析日之前最新一条作为当前认知，旧确认不被覆盖。
* 已确认场次占用后，若新资料显示对照款无货：必须补做对照或取消该场次，
  不能把原场次默认成已产生有效证据。
* 未确认场次可重排；已确认但并未实际改订的承诺不随预测移动。
* 同一场次可同时展示多款，但只有凑齐两款的可比条件才缩小缺口。
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field


def _d(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


@dataclass
class SessionPlan:
    window: dict
    bookings: list[dict] = field(default_factory=list)
    latest_availability: dict[str, dict] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)
    overcommitted: bool = False
    status: str = ""
    actions: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.window["id"]

    @property
    def layer(self) -> tuple[str, str]:
        return (self.window["segment"], self.window["slot"])

    @property
    def confirmed(self) -> bool:
        return any(b.get("confirmed_on") for b in self.bookings)

    @property
    def live_skus(self) -> list[str]:
        out = []
        for b in self.bookings:
            av = self.latest_availability.get(b["sku"])
            if av is None or av["available_units"] >= b["units"]:
                out.append(b["sku"])
        return sorted(set(out))

    @property
    def comparable_pair(self) -> bool:
        return len(self.live_skus) >= 2 and not self.conflicts

    @property
    def evidence_eligible(self) -> bool:
        """只有已确认且两款实际有货，场次才算能产出可比证据。"""
        return self.confirmed and self.comparable_pair

    def as_dict(self) -> dict:
        return {
            "window": self.id,
            "shop": self.window["shop"],
            "on": self.window["on"],
            "segment": self.window["segment"],
            "slot": self.window["slot"],
            "positions": self.window["positions"],
            "booked_skus": sorted({b["sku"] for b in self.bookings}),
            "live_skus": self.live_skus,
            "confirmed": self.confirmed,
            "confirmed_on": max(
                (b["confirmed_on"] for b in self.bookings if b.get("confirmed_on")),
                default=None,
            ),
            "conflicts": self.conflicts,
            "overcommitted": self.overcommitted,
            "comparable_pair": self.comparable_pair,
            "evidence_eligible": self.evidence_eligible,
            "status": self.status,
            "actions": self.actions,
        }


def plan(data: dict, as_of: str) -> list[SessionPlan]:
    """按分析日给出每场次状态、冲突与待办。"""
    cutoff = _d(as_of)
    latest: dict[tuple[str, str], dict] = {}
    for row in data["availability"]:
        known = _d(row["known_on"])
        if known is None or known > cutoff:
            continue
        key = (row["window"], row["sku"])
        if key not in latest or known >= _d(latest[key]["known_on"]):
            latest[key] = row

    by_window: dict[str, list[dict]] = defaultdict(list)
    for b in data["booking"]:
        confirmed = _d(b.get("confirmed_on"))
        if confirmed is not None and confirmed > cutoff:
            continue  # 分析日尚未生效的承诺
        by_window[b["window"]].append(b)

    plans = []
    for w in sorted(data["windows"], key=lambda r: (r["on"], r["id"])):
        p = SessionPlan(window=w, bookings=by_window.get(w["id"], []))
        units = 0
        for b in p.bookings:
            units += b["units"]
            av = latest.get((w["id"], b["sku"]))
            if av is not None:
                p.latest_availability[b["sku"]] = av
                if av["available_units"] < b["units"]:
                    p.conflicts.append(
                        {
                            "sku": b["sku"],
                            "booked": b["units"],
                            "available": av["available_units"],
                            "known_on": av["known_on"],
                        }
                    )
        p.overcommitted = units > w["positions"]
        missing = [b["sku"] for b in p.bookings if b["sku"] not in p.latest_availability]
        for sku in missing:
            p.conflicts.append(
                {
                    "sku": sku,
                    "booked": next(b["units"] for b in p.bookings if b["sku"] == sku),
                    "available": None,
                    "known_on": None,
                }
            )

        if p.conflicts:
            names = "、".join(c["sku"] for c in p.conflicts)
            if p.confirmed:
                p.status = "已确认场次对照款缺货"
                p.actions = [
                    f"为 {names} 补做对照款，或取消该场次",
                    "不得把该场次默认计为已产生有效证据",
                ]
            else:
                p.status = "未确认，需换对照款后再确认"
                p.actions = [f"为 {names} 更换对照款并重新确认"]
        elif not p.bookings:
            p.status = "未确认，可重排"
            p.actions = ["按本轮补采建议决定是否占用"]
        elif p.confirmed and p.comparable_pair:
            p.status = "已确认，可比"
            p.actions = []
        elif p.comparable_pair:
            p.status = "已配齐两款，待确认"
            p.actions = ["确认后占用场次"]
        elif p.confirmed:
            p.status = "已确认，但不足两款，无法形成可比"
            p.actions = ["补一款对照款"]
        else:
            p.status = "已占位，不足两款"
            p.actions = ["补一款对照款后再确认"]
        if p.overcommitted:
            p.actions.append(
                f"占位 {units} 超过场次名额 {w['positions']}，需调整"
            )
        plans.append(p)
    return plans


def suggestions(
    plans: list[SessionPlan], gaps: list[dict]
) -> list[dict]:
    """把目标层缺口映射到可操作的场次。

    名额按缺口的目标权重排序，而不是按‘观察最少’排序。
    推荐只落在未确认、无冲突、能同时安排两款的场次；已确认场次不自动移动。
    """
    gap_layers = {(g["segment"], g["slot"]) for g in gaps}
    weights = {(g["segment"], g["slot"]): g["mass"] for g in gaps}
    out = []
    for p in plans:
        if p.layer not in gap_layers:
            continue
        if p.confirmed:
            continue  # 已确认承诺不随预测移动
        if p.conflicts:
            continue
        out.append(
            {
                "window": p.id,
                "shop": p.window["shop"],
                "layer": f"{p.layer[0]}/{p.layer[1]}",
                "mass": weights.get(p.layer, 0.0),
                "on": p.window["on"],
                "positions": p.window["positions"],
                "status": p.status,
                "recommended": True,
                "needs": "同一场次同时安排目标款与对照款，两款都可订",
                "note": "同场两款都具备才缩小该层缺口；单款只增加曝光，不构成比较。",
            }
        )
    return out


def gap_backlog(plans: list[SessionPlan], gaps: list[dict]) -> list[dict]:
    """按目标权重列出缺口与可承接场次；无场次时标为需新增。"""
    by_layer: dict[tuple[str, str], list[SessionPlan]] = defaultdict(list)
    for p in plans:
        by_layer[p.layer].append(p)
    out = []
    for g in sorted(gaps, key=lambda r: (-r["mass"], r["segment"], r["slot"])):
        layer = (g["segment"], g["slot"])
        candidates = by_layer.get(layer, [])
        usable = [
            p
            for p in candidates
            if not p.confirmed and not p.conflicts and not p.overcommitted
        ]
        locked = [p for p in candidates if p.confirmed]
        out.append(
            {
                **g,
                "window_candidates": [p.id for p in usable],
                "locked_windows": [p.id for p in locked],
                "action": (
                    "在候选场次同时安排目标款与对照款"
                    if usable
                    else "没有可重排场次，需新增场次或等下一轮名额"
                ),
            }
        )
    return out
