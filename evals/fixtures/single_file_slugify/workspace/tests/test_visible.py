import unittest

from text_utils import slugify


class SlugifyVisibleTest(unittest.TestCase):
    def test_lowercases_and_joins_words(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_trims_outer_spaces(self):
        self.assertEqual(slugify("  Tiny Harness  "), "tiny-harness")


if __name__ == "__main__":
    unittest.main()
