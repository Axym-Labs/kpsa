import unittest

import torch

from kpsa.vision_prototype_strengthening_v8 import (
    construction_indices,
    source_rebuild_indices,
)


class VisionPrototypeStrengtheningV8Test(unittest.TestCase):
    def test_construction_indices_use_disjoint_offsets_per_class(self):
        represented = torch.tensor([0, 1, 2, 3, 250, 251, 252, 253])

        selected = construction_indices(represented)

        self.assertEqual(selected.tolist(), [20, 21, 22, 23, 270, 271, 272, 273])
        self.assertFalse(set(selected.tolist()) & set(represented.tolist()))

    def test_source_rebuild_indices_use_requested_class_local_block(self):
        represented = torch.tensor([0, 1, 2, 3, 250, 251, 252, 253])

        selected = source_rebuild_indices(represented, 11)

        self.assertEqual(selected.tolist(), [11, 12, 13, 14, 261, 262, 263, 264])


if __name__ == "__main__":
    unittest.main()
