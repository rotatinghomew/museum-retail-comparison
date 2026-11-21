"""核查练习数据的原始事实，不计算分析结果。"""
from pathlib import Path
import json
import unittest


class DrillDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'data' / 'drill.json'
        cls.data = json.loads(source.read_text(encoding='utf-8'))

    def test_schema(self):
        expected = {'visits': ('id', 'visitor', 'shop', 'slot', 'segment', 'sku', 'band', 'exposed_on', 'observed'), 'orders': ('id', 'visitor', 'sku', 'shop', 'sold_on', 'paid', 'refund_on'), 'listings': ('shop', 'sku', 'band', 'starts', 'ends', 'orderable'), 'target': ('shop', 'segment', 'slot', 'mass'), 'intake': ('as_of', 'purchase_days', 'refund_days'), 'windows': ('id', 'shop', 'on', 'segment', 'slot', 'positions'), 'booking': ('id', 'window', 'sku', 'units', 'confirmed_on'), 'availability': ('window', 'sku', 'available_units', 'known_on')}
        self.assertEqual(set(self.data), set(expected))
        for name, fields in expected.items():
            with self.subTest(table=name):
                self.assertTrue(self.data[name])
                self.assertTrue(all(set(row) == set(fields) for row in self.data[name]))

    def test_repeated_visits_and_cross_shop_orders(self):
        visits = [row for row in self.data['visits'] if row['visitor'] == 'v1']
        order = next(row for row in self.data['orders'] if row['visitor'] == 'v1')
        self.assertEqual(len({row['shop'] for row in visits}), 2)
        self.assertNotIn(order['shop'], {row['shop'] for row in visits})
        self.assertIsNotNone(order['refund_on'])

    def test_confirmed_pair_meets_availability_update(self):
        skus = {row['sku'] for row in self.data['booking'] if row['window'] == 'W1'}
        self.assertEqual(len(skus), 2)
        affected = [row for row in self.data['availability'] if row['window'] == 'W1' and row['available_units'] == 0]
        self.assertEqual(len(affected), 1)
        self.assertIn(affected[0]['sku'], skus)
        self.assertTrue(any(row['window'] == 'W2' for row in self.data['availability']))
