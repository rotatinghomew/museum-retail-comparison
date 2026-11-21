"""馆内试销 -> 合作门店的可比证据引擎。

只用标准库；数据为合成演练资料。

口径
----
* 曝光机会：门店 x 款式 x 价格档在 ``listings`` 可订期间内、``observed=True``、
  且不晚于分析日的到访记录。
* 归属：同一访客只取 (exposed_on, id) 最早的一条符合资格曝光；后续曝光、
  跨店成交、门店统计后撤单都跟随该曝光。
* 成交：首个曝光后 ``purchase_days`` 天内、同 SKU、能连到该访客的订单；
  到 ``refund_days`` 观察期结束仍未退款，才算净留单。观察期未到的订单
  单独计入 pending，既不进分子也不从成熟分母里消失。
* 匿名曝光没有连接键，只计覆盖缺口，不推测成交。
* 目标构成 ``target.mass`` 是合作方客群权重，不是馆内到访量。
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# --------------------------------------------------------------------------
# 基础
# --------------------------------------------------------------------------

def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _d(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


@dataclass(frozen=True)
class Config:
    """分析口径；默认阈值可在命令行覆盖。"""

    as_of: str = "2026-07-23"
    purchase_days: int = 4
    refund_days: int = 8
    min_mature: int = 8
    max_anon_share: float = 0.35
    max_pending_share: float = 0.10
    z: float = 1.96
    bootstrap: int = 20000
    seed: int = 20260723

    @classmethod
    def from_intake(cls, intake: list[dict], **overrides) -> "Config":
        # 后到资料可能追加新的成熟口径；取 as_of 最新的一条
        row = max(intake, key=lambda r: r["as_of"])
        base = dict(
            as_of=row["as_of"],
            purchase_days=int(row["purchase_days"]),
            refund_days=int(row["refund_days"]),
        )
        base.update(overrides)
        return cls(**base)

    @property
    def as_of_date(self) -> dt.date:
        return _d(self.as_of)


# --------------------------------------------------------------------------
# 客群层
# --------------------------------------------------------------------------

@dataclass
class Cohort:
    """SKU x 价格档 x (segment, slot) 的可比客群层。"""

    sku: str
    band: str
    segment: str
    slot: str
    shops: set[str] = field(default_factory=set)
    firsts: list[dict] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    pending_orders: dict[str, str] = field(default_factory=dict)
    anonymous: list[str] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    retained: dict[str, str] = field(default_factory=dict)
    refunded: dict[str, str] = field(default_factory=dict)
    order_shops: dict[str, str] = field(default_factory=dict)
    unmatched_orders: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.sku}|{self.band}|{self.segment}|{self.slot}"

    @property
    def label(self) -> str:
        return f"{self.sku} / {self.band} / {self.segment} / {self.slot}"

    @property
    def n_first(self) -> int:
        return len(self.firsts)

    @property
    def n_pending(self) -> int:
        return len(self.pending)

    @property
    def n_mature(self) -> int:
        return self.n_first - self.n_pending

    @property
    def n_retained(self) -> int:
        return len(self.retained)

    @property
    def n_refunded(self) -> int:
        return len(self.refunded)

    @property
    def n_anonymous(self) -> int:
        return len(self.anonymous)

    @property
    def anon_share(self) -> float:
        denom = self.n_first + self.n_anonymous
        return self.n_anonymous / denom if denom else 0.0

    @property
    def rate(self) -> float | None:
        return _ratio(self.n_retained, self.n_mature)

    @property
    def complete_observation(self) -> bool:
        return self.n_first > 0 and self.n_pending == 0

    def coverage_adequate(self, cfg: Config | None = None) -> bool:
        cfg = cfg or Config()
        return self.anon_share <= cfg.max_anon_share

    @property
    def pending_share(self) -> float:
        return self.n_pending / self.n_first if self.n_first else 0.0

    def reliable(self, cfg: Config | None = None) -> bool:
        """可进入定量讨论：成熟样本够、匿名缺口可接受、未到期比例小。

        未到期订单不进入分子，也不进成熟分母；比例小才允许先讨论，
        但仍标注观察未完全到期。
        """
        cfg = cfg or Config()
        return (
            self.n_mature >= cfg.min_mature
            and self.coverage_adequate(cfg)
            and self.pending_share <= cfg.max_pending_share
        )

    @property
    def zero_conversion(self) -> bool:
        """完整计数且零成交。不能被写成‘资料不足’。"""
        return self.n_mature > 0 and self.n_pending == 0 and self.n_retained == 0

    def status(self, cfg: Config | None = None) -> str:
        cfg = cfg or Config()
        if self.n_first == 0 and self.n_anonymous == 0:
            return "无曝光"
        if self.n_first == 0:
            return "仅匿名曝光，覆盖缺口"
        if self.n_pending == self.n_first:
            return "观察未到期"
        if self.n_pending:
            return "部分观察未到期"
        if self.n_retained == 0:
            if self.n_mature >= cfg.min_mature and self.coverage_adequate(cfg):
                return "零成交（观察完整）"
            return "零成交（观察完整，样本少）"
        if not self.coverage_adequate(cfg):
            return "有成交但匿名缺口大"
        if self.n_mature < cfg.min_mature:
            return "有成交但样本少"
        return "可比较"

    def ci(self, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson 区间（成熟样本为分母）。"""
        n, k = self.n_mature, self.n_retained
        if n == 0:
            return None
        p = k / n
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return (max(0.0, centre - half), min(1.0, centre + half))

    def as_dict(self, cfg: Config | None = None) -> dict:
        cfg = cfg or Config()
        return {
            "key": self.key,
            "label": self.label,
            "sku": self.sku,
            "band": self.band,
            "segment": self.segment,
            "slot": self.slot,
            "shops": sorted(self.shops),
            "first_exposures": self.n_first,
            "mature": self.n_mature,
            "pending_refund_window": self.n_pending,
            "retained": self.n_retained,
            "refunded": self.n_refunded,
            "rate": self.rate,
            "ci": self.ci(cfg.z),
            "anonymous": self.n_anonymous,
            "anon_share": round(self.anon_share, 4),
            "complete_observation": self.complete_observation,
            "zero_conversion": self.zero_conversion,
            "pending_share": round(self.pending_share, 4),
            "reliable": self.reliable(cfg),
            "status": self.status(cfg),
            "cross_shop_orders": sum(1 for s in self.order_shops.values() if s != "馆内"),
            "excluded_exposures": len(self.excluded),
        }


# --------------------------------------------------------------------------
# 归属
# --------------------------------------------------------------------------

def _eligible_exposure(v: dict, listings: list[dict], as_of: dt.date) -> bool:
    if not v.get("observed"):
        return False
    exp = _d(v["exposed_on"])
    if exp is None or exp > as_of:
        return False
    for row in listings:
        if (
            row["shop"] == v["shop"]
            and row["sku"] == v["sku"]
            and row["band"] == v["band"]
            and row["orderable"]
            and _d(row["starts"]) <= exp <= _d(row["ends"])
            and _d(row["starts"]) <= as_of
        ):
            return True
    return False


def collapse_orders(orders: list[dict]) -> list[dict]:
    """同一订单号若被后到流水再次追加，取信息更全的一行。

    退款流水通常以同 id、补上 ``refund_on`` 的形式追加；旧行仍保留在原始
    资料里，但分析只认最新状态，避免同一订单既算留单又算退款。
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    for o in orders:
        oid = o["id"]
        if oid not in best:
            order.append(oid)
            best[oid] = o
            continue
        cur = best[oid]
        # refund_on 有值、paid 更明确、sold_on 更晚 -> 视为更新
        if (o.get("refund_on") is not None and cur.get("refund_on") is None) or (
            o.get("refund_on") is None
            and cur.get("refund_on") is None
            and (o.get("sold_on") or "") >= (cur.get("sold_on") or "")
        ):
            best[oid] = o
    return [best[oid] for oid in order]


def _linked_orders(visitor: str, exposure: dict, orders: list[dict], cfg: Config) -> list[dict]:
    start = _d(exposure["exposed_on"])
    end = start + dt.timedelta(days=cfg.purchase_days)
    out = []
    for o in orders:
        if o["visitor"] != visitor or o["sku"] != exposure["sku"]:
            continue
        sold = _d(o["sold_on"])
        if sold is None or sold > cfg.as_of_date:
            continue
        if start <= sold <= end:
            out.append(o)
    return out


def _order_state(o: dict, cfg: Config) -> str:
    sold = _d(o["sold_on"])
    refund = _d(o["refund_on"])
    if refund is not None and refund <= cfg.as_of_date:
        return "refunded"
    if sold + dt.timedelta(days=cfg.refund_days) <= cfg.as_of_date:
        return "retained"
    return "pending"


def build_cohorts(data: dict, cfg: Config | None = None) -> dict[str, Cohort]:
    cfg = cfg or Config.from_intake(data["intake"])
    as_of = cfg.as_of_date
    cohorts: dict[str, Cohort] = {}
    first_by_visitor: dict[str, dict] = {}
    duplicates: dict[str, list[dict]] = defaultdict(list)
    orders = collapse_orders(data["orders"])

    def get(e: dict) -> Cohort:
        key = f"{e['sku']}|{e['band']}|{e['segment']}|{e['slot']}"
        if key not in cohorts:
            cohorts[key] = Cohort(
                sku=e["sku"], band=e["band"], segment=e["segment"], slot=e["slot"]
            )
        c = cohorts[key]
        c.shops.add(e["shop"])
        return c

    for v in sorted(data["visits"], key=lambda r: (r["exposed_on"], r["id"])):
        if not _eligible_exposure(v, data["listings"], as_of):
            continue
        c = get(v)
        if v["visitor"] is None:
            c.anonymous.append(v["id"])
            continue
        if v["visitor"] in first_by_visitor:
            duplicates[v["visitor"]].append(v)
            continue
        first_by_visitor[v["visitor"]] = v
        c.firsts.append(v)

    for visitor, extras in duplicates.items():
        for v in extras:
            cohorts[f"{v['sku']}|{v['band']}|{v['segment']}|{v['slot']}"].excluded.append(
                {"id": v["id"], "visitor": visitor, "reason": "同一访客已有更早符合资格曝光"}
            )

    for visitor, e in first_by_visitor.items():
        c = cohorts[f"{e['sku']}|{e['band']}|{e['segment']}|{e['slot']}"]
        linked = _linked_orders(visitor, e, orders, cfg)
        if not linked:
            # 没有成交：购买窗口未关闭才是 pending；关闭则是完整零成交
            if _d(e["exposed_on"]) + dt.timedelta(days=cfg.purchase_days) > as_of:
                c.pending.append(visitor)
            continue
        states = [(o, _order_state(o, cfg)) for o in linked]
        if any(state == "pending" for _, state in states):
            c.pending.append(visitor)
            c.pending_orders[visitor] = next(
                o["id"] for o, state in states if state == "pending"
            )
            continue
        kept = [o for o, state in states if state == "retained"]
        if kept:
            o = kept[0]
            c.retained[visitor] = o["id"]
            c.order_shops[o["id"]] = o["shop"]
        else:
            o = states[0][0]
            c.refunded[visitor] = o["id"]
            c.order_shops[o["id"]] = o["shop"]
    return cohorts


def duplicate_audit(data: dict, cfg: Config | None = None) -> list[dict]:
    """返回被首个曝光规则排除的后续曝光。"""
    cfg = cfg or Config.from_intake(data["intake"])
    seen: dict[str, dict] = {}
    out = []
    for v in sorted(data["visits"], key=lambda r: (r["exposed_on"], r["id"])):
        if not _eligible_exposure(v, data["listings"], cfg.as_of_date) or v["visitor"] is None:
            continue
        if v["visitor"] in seen:
            out.append(
                {
                    "visitor": v["visitor"],
                    "exposure": v["id"],
                    "shop": v["shop"],
                    "sku": v["sku"],
                    "band": v["band"],
                    "exposed_on": v["exposed_on"],
                    "attributed_to": seen[v["visitor"]]["id"],
                    "attributed_shop": seen[v["visitor"]]["shop"],
                    "attributed_on": seen[v["visitor"]]["exposed_on"],
                }
            )
        else:
            seen[v["visitor"]] = v
    return out


# --------------------------------------------------------------------------
# 目标构成
# --------------------------------------------------------------------------

def target_mix(data: dict, shop: str) -> dict[tuple[str, str], float]:
    rows = [r for r in data["target"] if r["shop"] == shop]
    total = sum(r["mass"] for r in rows)
    if not total:
        return {}
    return {(r["segment"], r["slot"]): r["mass"] / total for r in rows}


def _target_cells(
    cohorts: dict[str, Cohort],
    mix: dict[tuple[str, str], float],
    sku: str,
    band: str,
    cfg: Config,
) -> list[dict]:
    cells = []
    for (segment, slot), mass in sorted(mix.items()):
        if mass <= 0:
            continue
        key = f"{sku}|{band}|{segment}|{slot}"
        c = cohorts.get(key)
        covered = bool(c and c.n_mature > 0 and c.coverage_adequate(cfg))
        cells.append(
            {
                "segment": segment,
                "slot": slot,
                "mass": mass,
                "cohort": key if c else None,
                "mature": c.n_mature if c else 0,
                "pending": c.n_pending if c else 0,
                "retained": c.n_retained if c else 0,
                "rate": c.rate if c else None,
                "ci": c.ci(cfg.z) if c else None,
                "anonymous": c.n_anonymous if c else 0,
                "covered": covered,
                "reliable": bool(c and c.reliable(cfg)),
                "zero_conversion": bool(c and c.zero_conversion),
                "status": c.status(cfg) if c else "无曝光",
            }
        )
    return cells


def target_coverage(
    cohorts: dict[str, Cohort],
    mix: dict[tuple[str, str], float],
    sku: str,
    band: str,
    cfg: Config,
) -> dict:
    """目标层覆盖。占比>0 的层缺证据时只能列为缺口，不能拿别层填。"""
    cells = _target_cells(cohorts, mix, sku, band, cfg)
    covered_mass = sum(c["mass"] for c in cells if c["covered"])
    reliable_mass = sum(c["mass"] for c in cells if c["reliable"])
    gaps = [
        {"segment": c["segment"], "slot": c["slot"], "mass": c["mass"], "status": c["status"]}
        for c in cells
        if not c["covered"]
    ]
    weak = [
        {"segment": c["segment"], "slot": c["slot"], "mass": c["mass"], "status": c["status"]}
        for c in cells
        if c["covered"] and not c["reliable"]
    ]
    return {
        "sku": sku,
        "band": band,
        "cells": cells,
        "layers": [f"{c['segment']}/{c['slot']}" for c in cells],
        "covered_mass": covered_mass,
        "reliable_mass": reliable_mass,
        "gap_mass": 1.0 - covered_mass,
        "gaps": gaps,
        "weak_cells": weak,
        "complete": not gaps,
        "reliable_complete": not gaps and not weak,
        "zero_layers": [
            f"{c['segment']}/{c['slot']}" for c in cells if c["zero_conversion"]
        ],
    }


def weighted_estimate(
    cohorts: dict[str, Cohort],
    mix: dict[tuple[str, str], float],
    sku: str,
    band: str,
    cfg: Config,
) -> dict:
    """只有所有目标层都有观察时，才给整店加权估计。"""
    cov = target_coverage(cohorts, mix, sku, band, cfg)
    if not cov["complete"]:
        return {
            "sku": sku,
            "band": band,
            "estimate": None,
            "ci": None,
            "reason": "目标层存在覆盖缺口，不能给整店加权估计",
            "coverage": cov,
        }
    rate = sum(
        c["mass"] * cohorts[c["cohort"]].rate
        for c in cov["cells"]
        if c["cohort"] in cohorts and cohorts[c["cohort"]].rate is not None
    )
    ci = _weighted_bootstrap(cohorts, mix, sku, band, cfg)
    return {
        "sku": sku,
        "band": band,
        "estimate": rate,
        "ci": ci,
        "reason": "含未到期订单" if any(c["pending"] for c in cov["cells"]) else None,
        "coverage": cov,
    }


# --------------------------------------------------------------------------
# 不确定性与比较
# --------------------------------------------------------------------------

def _draw_binom(n: int, p: float, rng: random.Random) -> int:
    if n <= 0:
        return 0
    return sum(1 for _ in range(n) if rng.random() < p)


def _bootstrap_rates(
    cohorts: dict[str, Cohort],
    mix: dict[tuple[str, str], float],
    sku: str,
    band: str,
    layers: list[tuple[str, str]],
    cfg: Config,
    rng: random.Random,
) -> list[float]:
    out = []
    for _ in range(cfg.bootstrap):
        num = 0.0
        denom = 0.0
        for layer in layers:
            c = cohorts.get(f"{sku}|{band}|{layer[0]}|{layer[1]}")
            if not c or c.n_mature == 0:
                continue
            p = c.n_retained / c.n_mature
            k = _draw_binom(c.n_mature, p, rng)
            num += mix[layer] * (k / c.n_mature)
            denom += mix[layer]
        out.append(num / denom if denom else 0.0)
    return out


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _weighted_bootstrap(
    cohorts: dict[str, Cohort],
    mix: dict[tuple[str, str], float],
    sku: str,
    band: str,
    cfg: Config,
) -> tuple[float, float] | None:
    layers = [k for k in mix if mix[k] > 0]
    rng = random.Random(cfg.seed)
    vals = _bootstrap_rates(cohorts, mix, sku, band, layers, cfg, rng)
    if not vals:
        return None
    return (_pct(vals, 0.025), _pct(vals, 0.975))


def compare_common(
    cohorts: dict[str, Cohort],
    mix: dict[tuple[str, str], float],
    skus: Iterable[str],
    band: str,
    cfg: Config,
) -> dict:
    """只在所有款式都可信的共同层上比较，并报告覆盖与排序稳健性。"""
    skus = sorted(skus)
    positive = [k for k in mix if mix[k] > 0]
    common = []
    for layer in positive:
        cells = [cohorts.get(f"{sku}|{band}|{layer[0]}|{layer[1]}") for sku in skus]
        if cells and all(c and c.reliable(cfg) for c in cells):
            common.append(layer)
    covered_mass = sum(mix[k] for k in common)
    base = {
        "skus": skus,
        "band": band,
        "covered_mass": covered_mass,
        "common_layers": [f"{s}/{l}" for s, l in common],
        "uncovered_layers": [f"{s}/{l}" for s, l in positive if (s, l) not in common],
    }
    if len(skus) < 2:
        return {
            **base,
            "results": [],
            "ranking": [],
            "order_stable": None,
            "uncertainty_changes_ranking": None,
            "note": "只圈选了一个款式，不做款式间排序。",
        }
    if not common:
        blocked = []
        for layer in positive:
            missing = [
                sku
                for sku in skus
                if not (
                    cohorts.get(f"{sku}|{band}|{layer[0]}|{layer[1]}")
                    and cohorts[f"{sku}|{band}|{layer[0]}|{layer[1]}"].reliable(cfg)
                )
            ]
            blocked.append({"layer": f"{layer[0]}/{layer[1]}", "missing": missing})
        return {
            **base,
            "results": [],
            "ranking": [],
            "blocked_layers": blocked,
            "order_stable": None,
            "uncertainty_changes_ranking": None,
            "note": "没有所有款式都具备可信观察的共同层，不能排序；缺口见 blocked_layers。",
        }

    sub_mix = {k: mix[k] for k in common}
    results = []
    boot: dict[str, list[float]] = {}
    for sku in skus:
        rng = random.Random(cfg.seed + 1)
        vals = _bootstrap_rates(cohorts, sub_mix, sku, band, common, cfg, rng)
        boot[sku] = vals
        rate = (
            sum(
                mix[k] * cohorts[f"{sku}|{band}|{k[0]}|{k[1]}"].rate
                for k in common
            )
            / covered_mass
        )
        cells = [cohorts[f"{sku}|{band}|{k[0]}|{k[1]}"] for k in common]
        results.append(
            {
                "sku": sku,
                "rate": rate,
                "ci": (_pct(vals, 0.025), _pct(vals, 0.975)) if vals else None,
                "n_mature": sum(c.n_mature for c in cells),
                "retained": sum(c.n_retained for c in cells),
                "cells": [c.key for c in cells],
                "covered_mass": covered_mass,
            }
        )
    ranked = sorted(results, key=lambda r: r["rate"], reverse=True)

    order_stable = None
    diff_ci = None
    p_better = None
    if len(ranked) == 2:
        a, b = ranked[0]["sku"], ranked[1]["sku"]
        diffs = sorted(x - y for x, y in zip(boot[a], boot[b]))
        diff_ci = (_pct(diffs, 0.025), _pct(diffs, 0.975))
        p_better = sum(1 for d in diffs if d > 0) / len(diffs)
        order_stable = diff_ci[0] > 0
    return {
        **base,
        "results": results,
        "ranking": [r["sku"] for r in ranked],
        "pairwise_diff_ci": diff_ci,
        "probability_leader_better": p_better,
        "order_stable": order_stable,
        "uncertainty_changes_ranking": (not order_stable) if order_stable is not None else None,
        "note": (
            "只在共同可信层上比较；covered_mass 是这些层占目标机会的比例。"
            "排序不稳健时不得把先后当成结论。"
        ),
    }


# --------------------------------------------------------------------------
# 分店敏感性与记录覆盖
# --------------------------------------------------------------------------

def _visit_shop(data: dict, exposure_id: str) -> str | None:
    return next((v["shop"] for v in data["visits"] if v["id"] == exposure_id), None)


def per_shop_sensitivity(
    data: dict, cohorts: dict[str, Cohort], cfg: Config, band: str | None = None
) -> list[dict]:
    """按曝光门店拆分（归属仍跟随全局首个曝光）。"""
    rows: dict[tuple[str, str, str, str, str], dict] = {}

    def row_for(shop: str, c: Cohort) -> dict:
        key = (shop, c.sku, c.band, c.segment, c.slot)
        return rows.setdefault(
            key,
            {
                "shop": shop,
                "sku": c.sku,
                "band": c.band,
                "segment": c.segment,
                "slot": c.slot,
                "first": 0,
                "pending": 0,
                "retained": 0,
                "refunded": 0,
                "anonymous": 0,
            },
        )

    for c in cohorts.values():
        for e in c.firsts:
            r = row_for(e["shop"], c)
            r["first"] += 1
            if e["visitor"] in c.pending:
                r["pending"] += 1
            elif e["visitor"] in c.retained:
                r["retained"] += 1
            elif e["visitor"] in c.refunded:
                r["refunded"] += 1
        for aid in c.anonymous:
            shop = _visit_shop(data, aid)
            if shop:
                row_for(shop, c)["anonymous"] += 1

    out = []
    for (shop, sku, b, seg, slot), r in sorted(rows.items()):
        if band and b != band:
            continue
        mature = r["first"] - r["pending"]
        zero = mature > 0 and r["pending"] == 0 and r["retained"] == 0
        out.append(
            {
                **r,
                "mature": mature,
                "rate": _ratio(r["retained"], mature),
                "zero_conversion": zero,
                "status": (
                    "零成交（观察完整）"
                    if zero and mature >= cfg.min_mature
                    else "零成交（观察完整，样本少）"
                    if zero
                    else "部分观察未到期"
                    if r["pending"]
                    else "有成交"
                    if r["retained"]
                    else "无成熟观察"
                ),
            }
        )
    return out


def recording_density(
    data: dict, cohorts: dict[str, Cohort], cfg: Config, band: str | None = None
) -> list[dict]:
    """各门店 x 层的曝光天数与日均到访，提示‘只记录部分时段’的风险。"""
    acc: dict[tuple[str, str, str, str, str], dict] = {}
    for v in data["visits"]:
        if band and v["band"] != band:
            continue
        if _d(v["exposed_on"]) > cfg.as_of_date:
            continue
        key = (v["shop"], v["sku"], v["band"], v["segment"], v["slot"])
        r = acc.setdefault(
            key,
            {"shop": v["shop"], "sku": v["sku"], "band": v["band"],
             "segment": v["segment"], "slot": v["slot"],
             "visits": 0, "observed": 0, "dates": set()},
        )
        r["visits"] += 1
        r["observed"] += int(bool(v["observed"]))
        r["dates"].add(v["exposed_on"])
    out = []
    for r in acc.values():
        days = len(r.pop("dates"))
        out.append(
            {
                **r,
                "exposure_days": days,
                "visits_per_day": r["visits"] / days if days else None,
                "observed_share": r["observed"] / r["visits"] if r["visits"] else None,
                "low_density": days < 5 or (r["visits"] / days < 1.0 if days else True),
            }
        )
    return sorted(out, key=lambda r: (r["shop"], r["sku"], r["segment"], r["slot"]))


def heterogeneity(
    sensitivity: list[dict], layers: list[tuple[str, str]], skus: list[str]
) -> dict:
    """同层跨店率差，以及款式排序是否在门店间反向。"""
    by_key: dict[tuple[str, str, str, str], dict] = {}
    for r in sensitivity:
        by_key[(r["sku"], r["segment"], r["slot"], r["shop"])] = r

    layer_rows = []
    for seg, slot in layers:
        for sku in skus:
            rates = [
                (r["shop"], r["rate"], r["mature"])
                for (s, sg, sl, shop), r in by_key.items()
                if s == sku and sg == seg and sl == slot and r["mature"] > 0
            ]
            if not rates:
                continue
            vals = [x[1] for x in rates]
            spread = max(vals) - min(vals)
            ratio = (max(vals) / min(vals)) if min(vals) > 0 else None
            layer_rows.append(
                {
                    "sku": sku,
                    "layer": f"{seg}/{slot}",
                    "shops": [{"shop": s, "rate": v, "mature": m} for s, v, m in rates],
                    "spread": spread,
                    "ratio": ratio,
                    "severe": spread >= 0.20 or (ratio is not None and ratio >= 2.5),
                }
            )

    reversals = []
    if len(skus) == 2:
        a, b = skus
        for seg, slot in layers:
            signs = {}
            for shop in sorted({r["shop"] for r in sensitivity}):
                ra = by_key.get((a, seg, slot, shop))
                rb = by_key.get((b, seg, slot, shop))
                if ra and rb and ra["mature"] > 0 and rb["mature"] > 0:
                    diff = ra["rate"] - rb["rate"]
                    signs[shop] = (diff, ra["rate"], rb["rate"])
            if len(signs) >= 2 and any(d > 0 for d, _, _ in signs.values()) and any(
                d < 0 for d, _, _ in signs.values()
            ):
                reversals.append(
                    {
                        "layer": f"{seg}/{slot}",
                        "skus": [a, b],
                        "by_shop": {
                            shop: {"diff": d, a: ra, b: rb}
                            for shop, (d, ra, rb) in signs.items()
                        },
                    }
                )
    return {
        "layers": layer_rows,
        "severe_layers": [r for r in layer_rows if r["severe"]],
        "ranking_reversals": reversals,
        "note": (
            "跨店率差过大或排序反向时，合并率不能代表单一渠道；"
            "不得把分店差异当成款式优劣。"
        ),
    }


# --------------------------------------------------------------------------
# 审计
# --------------------------------------------------------------------------

def audit(data: dict, cohorts: dict[str, Cohort], cfg: Config) -> dict:
    orders = collapse_orders(data["orders"])
    linked: set[str] = set()
    for c in cohorts.values():
        linked |= set(c.retained.values()) | set(c.refunded.values())
        linked |= set(c.pending_orders.values())
        linked |= set(c.order_shops)
    unmatched = [o for o in orders if o["id"] not in linked]
    pending_orders = [
        {
            "id": o["id"],
            "visitor": o["visitor"],
            "sold_on": o["sold_on"],
            "refund_due": (_d(o["sold_on"]) + dt.timedelta(days=cfg.refund_days)).isoformat(),
        }
        for o in orders
        if o["id"] in linked and _order_state(o, cfg) == "pending"
    ]
    return {
        "visits_total": len(data["visits"]),
        "visits_observed": sum(1 for v in data["visits"] if v["observed"]),
        "visits_unobserved": sum(1 for v in data["visits"] if not v["observed"]),
        "anonymous_observed": sum(
            1 for v in data["visits"] if v["visitor"] is None and v["observed"]
        ),
        "first_exposures": sum(c.n_first for c in cohorts.values()),
        "pending_visitors": sum(c.n_pending for c in cohorts.values()),
        "retained_visitors": sum(c.n_retained for c in cohorts.values()),
        "refunded_visitors": sum(c.n_refunded for c in cohorts.values()),
        "orders_total": len(orders),
        "orders_raw": len(data["orders"]),
        "orders_linked": len(orders) - len(unmatched),
        "orders_unmatched": [
            {
                "id": o["id"],
                "visitor": o["visitor"],
                "sku": o["sku"],
                "sold_on": o["sold_on"],
                "shop": o["shop"],
                "reason": _unmatched_reason(o, data, cfg),
            }
            for o in unmatched
        ],
        "orders_pending_refund": pending_orders,
        "duplicates": duplicate_audit(data, cfg),
    }


def _unmatched_reason(o: dict, data: dict, cfg: Config) -> str:
    visit = next(
        (
            v
            for v in sorted(data["visits"], key=lambda r: (r["exposed_on"], r["id"]))
            if v["visitor"] == o["visitor"] and _eligible_exposure(v, data["listings"], cfg.as_of_date)
        ),
        None,
    )
    if visit is None:
        return "访客没有符合资格的曝光（窗口不完整/匿名/不可订/晚于分析日）"
    if o["sku"] != visit["sku"]:
        return "订单 SKU 与首个曝光不同"
    sold = _d(o["sold_on"])
    start = _d(visit["exposed_on"])
    if sold is None or sold > cfg.as_of_date:
        return "订单售出晚于分析日"
    if sold < start:
        return "订单早于首个曝光"
    if sold > start + dt.timedelta(days=cfg.purchase_days):
        return "订单超出购买归因窗口"
    return "未归类"


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    skus: tuple[str, ...] = ()
    band: str | None = None
    shops: tuple[str, ...] = ()


def _decision(
    coverage: dict,
    weighted: dict,
    comparison: dict,
    hetero: dict,
    sensitivity: list[dict],
    sku: str,
    band: str,
) -> dict:
    gaps = coverage["gaps"]
    zero_layers = coverage["zero_layers"]
    reliable_cells = [c for c in coverage["cells"] if c["reliable"]]
    covered_layers = [f"{c['segment']}/{c['slot']}" for c in reliable_cells]
    gap_layers = [f"{g['segment']}/{g['slot']}" for g in gaps]
    weak_layers = [f"{c['segment']}/{c['slot']}" for c in coverage["weak_cells"]]
    severe = [h["layer"] for h in hetero["severe_layers"] if h["sku"] == sku]
    reversal = [r["layer"] for r in hetero["ranking_reversals"] if sku in r["skus"]]
    zero_shops = [
        f"{r['shop']}/{r['segment']}/{r['slot']}"
        for r in sensitivity
        if r["sku"] == sku and r["band"] == band and r["zero_conversion"]
    ]
    order_stable = comparison.get("order_stable")

    reasons: list[str] = []
    total_mass = sum(c["mass"] for c in coverage["cells"] if c["mass"] > 0)
    if coverage["cells"] == [] or total_mass <= 0:
        category = "无法判断（缺少目标客群构成）"
        reasons.append("合作方目标构成缺失，不能把馆内结构当作目标结构。")
    elif not reliable_cells:
        if zero_layers:
            category = "仅馆内继续卖（目标层完整计数零成交）"
            reasons.append("目标层有完整计数但零成交，不能当作资料不足，也不能拿去谈铺货。")
        else:
            category = "暂不进入谈判（目标层无可信观察）"
            reasons.append("目标层没有可信观察，先补观察或留在馆内。")
    elif coverage["reliable_complete"]:
        if len(zero_layers) == len([c for c in coverage["cells"] if c["mass"] > 0]):
            category = "仅馆内继续卖（目标层完整计数零成交）"
            reasons.append("所有目标层完整计数且零成交。")
        elif zero_layers:
            category = "部分目标层完整计数零成交，不可作为整店可铺依据"
            reasons.append(
                f"零成交层 {', '.join(zero_layers)} 是完整计数结果，不随其他层改变。"
            )
        elif severe or reversal:
            category = "可按层谈试点，但不可承诺整店转化"
            reasons.append("目标层齐全，但跨店率差过大或排序反向，合并率不能代表合作门店。")
        elif any(c["pending"] for c in coverage["cells"]):
            category = "可谈整店试点（部分订单未到退款观察期）"
            reasons.append("目标层齐全；未到期订单未计入净留单，需在修订中复核。")
        else:
            category = "可谈整店试点"
            reasons.append("目标层齐全且无重大跨店冲突。")
    elif gap_layers:
        category = "仅可谈已覆盖层；缺口层继续馆内观察"
        reasons.append(
            f"已覆盖 {', '.join(covered_layers) or '无'}；缺口 {', '.join(gap_layers)}。"
        )
        reasons.append(
            "缺口层占比不为零，不能靠别的层填满，也不能删掉后声称是整店结论。"
        )
    else:
        category = "仅可谈已覆盖层；样本不足以支撑整店结论"
        reasons.append(f"已覆盖 {', '.join(covered_layers)}，但样本偏少，不能合并成整店结论。")
    if zero_layers:
        reasons.append(f"零成交层：{', '.join(zero_layers)}（完整计数，不是资料不足）。")
    if zero_shops:
        reasons.append(
            f"分店完整计数零成交：{', '.join(zero_shops)}（不是资料不足）。"
        )
    if weak_layers:
        reasons.append(f"样本偏少层：{', '.join(weak_layers)}。")
    if severe:
        reasons.append(f"跨店率差异大的层：{', '.join(severe)}。")
    if order_stable is False:
        reasons.append("共同层上的排序在不确定性下不稳健，不能把先后当结论。")

    return {
        "sku": sku,
        "band": band,
        "category": category,
        "worth_discussing": bool(reliable_cells)
        and total_mass > 0
        and len(zero_layers) != len([c for c in coverage["cells"] if c["mass"] > 0]),
        "whole_store_claim_allowed": (
            coverage["reliable_complete"] and not zero_layers and not severe and not reversal
        ),
        "ranking_claim_allowed": bool(order_stable) and not reversal,
        "covered_layers": covered_layers,
        "gap_layers": gap_layers,
        "zero_layers": zero_layers,
        "zero_conversion_shops": zero_shops,
        "weighted_rate": weighted.get("estimate"),
        "weighted_ci": weighted.get("ci"),
        "weighted_note": (
            "合并率仅供参考，不代表任一门店"
            if (severe or reversal) and weighted.get("estimate") is not None
            else None
        ),
        "reasons": reasons,
    }


def build_analysis(data: dict, cfg: Config | None = None, selection: Selection | None = None) -> dict:
    cfg = cfg or Config.from_intake(data["intake"])
    selection = selection or Selection()
    cohorts = build_cohorts(data, cfg)

    all_skus = sorted({r["sku"] for r in data["listings"]})
    skus = sorted(selection.skus) if selection.skus else all_skus
    all_bands = sorted({r["band"] for r in data["listings"]})
    bands = [selection.band] if selection.band else all_bands
    shops = list(selection.shops) if selection.shops else sorted(
        {r["shop"] for r in data["target"]}
    )

    assessments = []
    for shop in shops:
        mix = target_mix(data, shop)
        for band in bands:
            coverage = {
                sku: target_coverage(cohorts, mix, sku, band, cfg) for sku in skus
            }
            weighted = {
                sku: weighted_estimate(cohorts, mix, sku, band, cfg) for sku in skus
            }
            comparison = compare_common(cohorts, mix, skus, band, cfg)
            sens = per_shop_sensitivity(data, cohorts, cfg, band)
            hetero = heterogeneity(sens, list(mix.keys()), skus)
            assessments.append(
                {
                    "shop": shop,
                    "band": band,
                    "target_mix": {f"{k[0]}/{k[1]}": v for k, v in mix.items()},
                    "coverage": coverage,
                    "weighted": weighted,
                    "comparison": comparison,
                    "decisions": [
                        _decision(
                            coverage[sku], weighted[sku], comparison, hetero, sens, sku, band
                        )
                        for sku in skus
                    ],
                    "heterogeneity": hetero,
                    "sensitivity_by_shop": [
                        r for r in sens if r["sku"] in skus and r["band"] == band
                    ],
                    "recording_density": [
                        r for r in recording_density(data, cohorts, cfg, band)
                        if r["sku"] in skus
                    ],
                }
            )

    return {
        "scope": {"skus": skus, "bands": bands, "shops": shops},
        "cohorts": {k: v.as_dict(cfg) for k, v in sorted(cohorts.items())},
        "assessments": assessments,
        "audit": audit(data, cohorts, cfg),
    }
