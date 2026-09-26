import unittest

from kpsa.vision_family_refined import adaptive_family_ids


class VisionFamilyRefinedTest(unittest.TestCase):
    def test_adaptive_families_form_bounded_non_singleton_groups(self):
        paths = [
            (0, 1, 10),
            (0, 1, 11),
            (0, 2, 20),
            (0, 2, 21),
            (0, 2, 22),
        ]
        self.assertEqual(adaptive_family_ids(paths, maximum_size=2), [1, 1, 20, 21, 22])
        self.assertEqual(adaptive_family_ids(paths, maximum_size=3), [1, 1, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
