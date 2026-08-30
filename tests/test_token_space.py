import unittest

import torch

from anytrain.module.token_space import Layout


class TokenSpaceTest(unittest.TestCase):
    def test_layout_maps_named_blocks(self):
        layout = Layout(text=(0, 3), audio=(10, 12))
        values = torch.tensor([0, 2])
        self.assertTrue(torch.equal(layout.to_global("text", values), torch.tensor([0, 2])))
        self.assertTrue(torch.equal(layout.to_local(torch.tensor([10, 11])), torch.tensor([0, 1])))


if __name__ == "__main__":
    unittest.main()
