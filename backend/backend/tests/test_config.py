import os
import unittest

from app.config import get_dist_dir


class GetDistDirTests(unittest.TestCase):
    def test_get_dist_dir_prefers_frontend_build_output(self):
        candidates = [
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist-app")
            ),
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
            ),
        ]
        self.assertIn(get_dist_dir(), candidates)


if __name__ == "__main__":
    unittest.main()
