import unittest
from effectiveness import engine, benchmark


class TestEffectiveness(unittest.TestCase):
    def test_clear_effect_closes(self):
        r = engine.verify([6, 7, 5, 8, 6, 5, 7, 6, 9, 5, 6, 7], [1, 1, 2, 1, 0, 1, 2, 1])
        self.assertEqual(r["decision"], "CLOSE")

    def test_no_effect_not_closed(self):
        r = engine.verify([6, 7, 5, 8, 6, 5, 7, 6, 9, 5, 6, 7], [6, 7, 5, 8, 6, 7, 6, 5, 7, 6])
        self.assertIn(r["decision"], ("REOPEN", "EXTEND"))

    def test_too_early_monitors(self):
        self.assertEqual(engine.verify([6] * 12, [1, 1])["decision"], "MONITOR")

    def test_benchmark_false_closure_order(self):
        _, s = benchmark.run(reps=400)
        self.assertEqual(len(s), 5)
        f = s.set_index("method").false_closure
        self.assertLess(f["Bayesian sequential (proposed)"], f["Fixed 30 days: point estimate"])


if __name__ == "__main__":
    unittest.main()
