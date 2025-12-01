import unittest
from importlib import metadata

from auxin_signalling_models.version_info import get_version_info


class TestGetVersionInfo(unittest.TestCase):
    def test_get_version_info(self):
        info_msg = get_version_info()

        test_msg = metadata.metadata("auxin_signalling_models")

        for (k, v) in test_msg.items():
            self.assertIn(k, info_msg)

            found = False
            for v2 in info_msg.get_all(k):
                if v2 == v:
                    found = True
            self.assertTrue(found, f"Coudn't find field {k}: {v} in info")

        return
