"""核查分析口径：归属、成熟、覆盖与场次安排。"""
from pathlib import Path
import datetime as dt
import json
import tempfile
import unittest

from trial.booking import gap_backlog, plan, suggestions
from trial.engine import (
    Config,
    Selection,
    build_analysis,
    build_cohorts,
    duplicate_audit,
    load,
    per_shop_sensitivity,
)
from trial.review import build_report, merge, quote, revisions, write_snapshot

ROOT = Path(__file__).resolve().parents[1]
DATA = load(ROOT / "data" / "drill.json")
CFG = Config.from_intake(DATA["intake"])
SEL = Selection(skus=("纸域-v1", "陶语-v1"), band="59档", shops=("合作南店",))


class AttributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohorts = build_cohorts(DATA, CFG)

    def test_first_eligible_exposure_wins(self):
        dup = duplicate_audit(DATA, CFG)
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0]["visitor"], "v1")
        self.assertEqual(dup[0]["exposure"], "e96")
        self.assertEqual(dup[0]["attributed_to"], "e1")
        # 后续曝光不进分母
        c = self.cohorts["纸域-v1|59档|游客|工作日"]
        self.assertNotIn("e96", [e["id"] for e in c.firsts])

    def test_cross_shop_order_follows_first_exposure(self):
        c = self.cohorts["纸域-v1|59档|游客|工作日"]
        # v1 在馆东曝光，成交发生在网店；仍归入该曝光
        self.assertEqual(c.refunded.get("v1"), "o1")
        self.assertGreater(c.order_shops["o1"] != "馆内", 0)

    def test_unobserved_visit_and_order_not_counted(self):
        audit = build_analysis(DATA, CFG, SEL)["audit"]
        unmatched = {o["id"]: o["reason"] for o in audit["orders_unmatched"]}
        self.assertIn("o91", unmatched)   # v91 曝光窗口不完整
        self.assertIn("o26", unmatched)   # v26 曝光窗口不完整
        self.assertNotIn("o97", unmatched)  # later 可连接，但退款未到期

    def test_pending_refund_is_not_retained(self):
        c = self.cohorts["陶语-v1|59档|游客|工作日"]
        self.assertIn("later", c.pending)
        self.assertNotIn("later", c.retained)
        self.assertEqual(c.n_pending, 1)  # later；v91 曝光窗口不完整，不入组
        # 未到期订单不进分子，也不进成熟分母
        self.assertEqual(c.n_mature, c.n_first - c.n_pending)

    def test_anonymous_not_imputed(self):
        c = self.cohorts["纸域-v1|59档|游客|工作日"]
        self.assertEqual(c.n_anonymous, 4)
        # 匿名曝光既不加分母也不推测成交
        self.assertEqual(c.n_mature + c.n_pending, c.n_first)


class NoExposureTest(unittest.TestCase):
    def test_complete_zero_conversion_is_not_insufficient(self):
        sens = per_shop_sensitivity(DATA, build_cohorts(DATA, CFG), CFG, "59档")
        row = next(
            r for r in sens
            if r["shop"] == "馆西" and r["sku"] == "纸域-v1"
            and r["segment"] == "常客" and r["slot"] == "休息日"
        )
        self.assertEqual(row["mature"], 7)
        self.assertTrue(row["zero_conversion"])
        self.assertEqual(row["status"], "零成交（观察完整，样本少）")

    def test_target_gap_cannot_be_filled_by_other_layer(self):
        analysis = build_analysis(DATA, CFG, SEL)
        a = analysis["assessments"][0]
        cov = a["coverage"]["陶语-v1"]
        self.assertFalse(cov["complete"])
        self.assertAlmostEqual(cov["gap_mass"], 0.45)
        self.assertEqual(
            [(g["segment"], g["slot"]) for g in cov["gaps"]],
            [("常客", "休息日")],
        )
        decision = next(d for d in a["decisions"] if d["sku"] == "陶语-v1")
        self.assertFalse(decision["whole_store_claim_allowed"])
        self.assertIsNone(decision["weighted_rate"])

    def test_common_comparison_reports_coverage_and_uncertainty(self):
        cmp = build_analysis(DATA, CFG, SEL)["assessments"][0]["comparison"]
        self.assertAlmostEqual(cmp["covered_mass"], 0.55)
        self.assertEqual(cmp["common_layers"], ["游客/工作日"])
        self.assertFalse(cmp["order_stable"])
        self.assertTrue(cmp["uncertainty_changes_ranking"])


class BookingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = {p.id: p for p in plan(DATA, CFG.as_of)}

    def test_confirmed_window_with_missing_control_is_not_evidence(self):
        w1 = self.plans["W1"]
        self.assertTrue(w1.confirmed)
        self.assertFalse(w1.evidence_eligible)
        self.assertEqual(w1.conflicts[0]["sku"], "陶语-v1")
        self.assertIn("补做对照款", w1.actions[0])
        self.assertIn("不得把该场次默认计为已产生有效证据", w1.actions[1])

    def test_confirmed_commitment_does_not_move(self):
        gaps = [{"segment": "常客", "slot": "休息日", "mass": 0.45}]
        rec = suggestions(list(self.plans.values()), gaps)
        self.assertEqual([r["window"] for r in rec], ["W2"])

    def test_only_layers_with_gaps_get_windows(self):
        gaps = [{"segment": "常客", "slot": "休息日", "mass": 0.45}]
        rec = suggestions(list(self.plans.values()), gaps)
        self.assertNotIn("W3", [r["window"] for r in rec])

    def test_backlog_points_at_reorderable_window(self):
        gaps = [{"segment": "常客", "slot": "休息日", "mass": 0.45}]
        backlog = gap_backlog(list(self.plans.values()), gaps)
        self.assertEqual(backlog[0]["window_candidates"], ["W2"])
        self.assertEqual(backlog[0]["locked_windows"], ["W1"])


class ReviewTest(unittest.TestCase):
    def test_snapshot_is_immutable_and_revisioned(self):
        report = build_report(DATA, CFG, SEL)
        with tempfile.TemporaryDirectory() as tmp:
            first = write_snapshot(report, tmp)
            second = write_snapshot(report, tmp)
            self.assertNotEqual(first, second)
            self.assertIn("-r2", second.name)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 2)

    def test_later_intake_row_sets_maturity(self):
        data = json.loads(json.dumps(DATA))
        data["intake"].append(
            {"as_of": "2026-07-30", "purchase_days": 4, "refund_days": 8}
        )
        self.assertEqual(Config.from_intake(data["intake"]).as_of, "2026-07-30")

    def test_merge_appends_without_dropping_old_rows(self):
        extra = {
            "orders": [
                {"id": "o97", "visitor": "later", "sku": "陶语-v1", "shop": "网店",
                 "sold_on": "2026-07-21", "paid": 5900, "refund_on": "2026-07-22"}
            ]
        }
        merged = merge(DATA, extra)
        ids = [o["id"] for o in merged["orders"]]
        self.assertEqual(ids.count("o97"), 2)  # 退款确认是新追加，不覆盖旧行
        self.assertEqual(len(merged["orders"]), len(DATA["orders"]) + 1)

    def test_quoted_snapshot_stays_separate(self):
        report = build_report(DATA, CFG, SEL)
        with tempfile.TemporaryDirectory() as tmp:
            first = write_snapshot(report, tmp)
            quote(first, "与门店沟通")
            second = write_snapshot(build_report(DATA, CFG, SEL), tmp)
            rows = revisions(tmp)
            self.assertTrue(rows[0]["quoted"])
            self.assertFalse(rows[1]["quoted"])
            self.assertEqual(rows[0]["quoted_note"], "与门店沟通")
            self.assertNotEqual(first, second)

    def test_new_refund_data_reclassifies(self):
        # 7/23 时 o97 仍在退款观察期；追加退款确认后应改为退款
        data = json.loads(json.dumps(DATA))
        cohorts = build_cohorts(data, CFG)
        c = cohorts["陶语-v1|59档|游客|工作日"]
        self.assertIn("later", c.pending)
        self.assertNotIn("later", c.retained)

        extra = {
            "orders": [
                {"id": "o97", "visitor": "later", "sku": "陶语-v1", "shop": "网店",
                 "sold_on": "2026-07-21", "paid": 5900, "refund_on": "2026-07-22"}
            ]
        }
        merged = merge(data, extra)
        c2 = build_cohorts(merged, CFG)["陶语-v1|59档|游客|工作日"]
        self.assertIn("later", c2.refunded)
        self.assertNotIn("later", c2.retained)


if __name__ == "__main__":
    unittest.main()


def make_rows(
    n, sku, band, segment, slot, shop="馆东", start="2026-07-01",
    observed=True, visitor_prefix="z", orders=None,
):
    """生成最小合成资料，用于口径边界测试。"""
    d = dt.date.fromisoformat(start)
    visits, ords = [], []
    for i in range(n):
        vid = f"{visitor_prefix}{i}"
        visits.append(
            {
                "id": f"x{i}", "visitor": vid, "shop": shop, "slot": slot,
                "segment": segment, "sku": sku, "band": band,
                "exposed_on": d.isoformat(), "observed": observed,
            }
        )
        if orders and i in orders:
            ords.append(
                {
                    "id": f"ox{i}", "visitor": vid, "sku": sku, "shop": "网店",
                    "sold_on": (d + dt.timedelta(days=1)).isoformat(),
                    "paid": 100, "refund_on": orders[i],
                }
            )
    return visits, ords


class ZeroAndMaturityTest(unittest.TestCase):
    def _data(self, visits, orders, mass):
        return {
            "visits": visits,
            "orders": orders,
            "listings": [
                {"shop": "馆东", "sku": "k", "band": "b", "starts": "2026-07-01",
                 "ends": "2026-07-31", "orderable": True}
            ],
            "target": [{"shop": "S", "segment": "常客", "slot": "休息日", "mass": mass},
                       {"shop": "S", "segment": "游客", "slot": "工作日", "mass": 100 - mass}],
            "intake": [{"as_of": "2026-07-23", "purchase_days": 4, "refund_days": 8}],
            "windows": [], "booking": [], "availability": [],
        }

    def test_full_zero_target_layer_is_sell_in_store_only(self):
        v1, o1 = make_rows(12, "k", "b", "常客", "休息日")
        v2, o2 = make_rows(12, "k", "b", "游客", "工作日", visitor_prefix="y")
        data = self._data(v1 + v2, o1 + o2, mass=50)
        report = build_report(data, Config.from_intake(data["intake"]),
                              Selection(skus=("k",), band="b", shops=("S",)))
        dec = report["assessments"][0]["decisions"][0]
        self.assertEqual(dec["category"], "仅馆内继续卖（目标层完整计数零成交）")
        self.assertFalse(dec["worth_discussing"])
        self.assertFalse(dec["whole_store_claim_allowed"])
        zero = [q for q in report["quality"] if q["code"] == "complete_zero"]
        self.assertTrue(zero)
        self.assertIn("不是资料不足", zero[0]["message"])

    def test_small_layer_is_insufficient_not_zero(self):
        # 5 次完整零成交目标层：样本少，但仍不能报成资料不足
        v1, o1 = make_rows(5, "k", "b", "常客", "休息日")
        v2, o2 = make_rows(12, "k", "b", "游客", "工作日", visitor_prefix="y")
        data = self._data(v1 + v2, o1 + o2, mass=50)
        report = build_report(data, Config.from_intake(data["intake"]),
                              Selection(skus=("k",), band="b", shops=("S",)))
        c = report["cohorts"]["k|b|常客|休息日"]
        self.assertEqual(c["status"], "零成交（观察完整，样本少）")
        self.assertTrue(c["zero_conversion"])
        self.assertFalse(c["reliable"])

    def test_pending_only_layer_is_not_zero(self):
        v1, o1 = make_rows(12, "k", "b", "常客", "休息日",
                           start="2026-07-20", orders={0: None})
        v2, o2 = make_rows(12, "k", "b", "游客", "工作日", visitor_prefix="y")
        data = self._data(v1 + v2, o1 + o2, mass=50)
        report = build_report(data, Config.from_intake(data["intake"]),
                              Selection(skus=("k",), band="b", shops=("S",)))
        c = report["cohorts"]["k|b|常客|休息日"]
        self.assertEqual(c["pending_refund_window"], 12)
        self.assertFalse(c["zero_conversion"])
        self.assertEqual(c["status"], "观察未到期")
        self.assertFalse(c["reliable"])
