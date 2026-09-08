import json
import unittest

from tiny_harness.context.token_meter import HeuristicTokenMeter


class HeuristicTokenMeterTest(unittest.TestCase):
    def test_estimates_complete_compact_json_context_at_four_chars_per_token(self):
        messages = [{"role": "system", "content": "rules"}]
        tools = [{"type": "function", "function": {"name": "read"}}]
        serialized = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        estimated = HeuristicTokenMeter().estimate(messages, tools)

        self.assertEqual(estimated, (len(serialized) + 3) // 4)

    def test_rounds_nonempty_partial_tokens_up(self):
        meter = HeuristicTokenMeter()

        estimated = meter.estimate([], [])

        self.assertGreater(estimated, 0)


if __name__ == "__main__":
    unittest.main()
