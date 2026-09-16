"""The sample images must survive a git checkout byte-exact.

A PPM begins with an ASCII header followed by raw pixel bytes, so git's text
heuristic can misclassify it and rewrite line endings on a Windows checkout.
That silently alters pixel data: it once changed a sample from "Speed limit
(30km/h)" to "Roundabout mandatory". `.gitattributes` marks these binary; these
tests fail if that protection is ever lost.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class TestSampleImages(unittest.TestCase):
    def test_samples_exist(self):
        files = sorted(SAMPLES.glob("*.ppm"))
        self.assertGreater(len(files), 0, "no sample images shipped")

    def test_no_crlf_in_pixel_data(self):
        """The signature of line-ending corruption."""
        for f in sorted(SAMPLES.glob("*.ppm")):
            data = f.read_bytes()
            self.assertEqual(
                data.count(b"\r\n"), 0,
                f"{f.name} contains CRLF -- git converted a binary file. "
                "Check that .gitattributes marks *.ppm binary.",
            )

    def test_header_is_wellformed(self):
        """P6 magic, then width height maxval, then binary pixels."""
        for f in sorted(SAMPLES.glob("*.ppm")):
            data = f.read_bytes()
            self.assertTrue(data.startswith(b"P6"), f"{f.name} is not a binary PPM")

    def test_pixel_payload_matches_declared_size(self):
        """Injected or stripped bytes change the payload length.

        Catches corruption even when it happens to avoid leaving CRLF behind.
        """
        for f in sorted(SAMPLES.glob("*.ppm")):
            data = f.read_bytes()
            fields, pos = [], 2
            while len(fields) < 3 and pos < len(data):
                while pos < len(data) and data[pos : pos + 1].isspace():
                    pos += 1
                if data[pos : pos + 1] == b"#":           # comment line
                    while pos < len(data) and data[pos] != 0x0A:
                        pos += 1
                    continue
                start = pos
                while pos < len(data) and not data[pos : pos + 1].isspace():
                    pos += 1
                fields.append(int(data[start:pos]))
            pos += 1  # the single whitespace byte after maxval

            width, height, _ = fields
            expected = width * height * 3
            self.assertEqual(
                len(data) - pos, expected,
                f"{f.name}: pixel payload is {len(data) - pos} bytes, "
                f"expected {expected} for {width}x{height} RGB",
            )

    def test_gitattributes_protects_binaries(self):
        ga = ROOT / ".gitattributes"
        self.assertTrue(ga.exists(), ".gitattributes is missing")
        text = ga.read_text(encoding="utf-8")
        for pattern in ("*.ppm", "*.pt", "*.png"):
            self.assertIn(pattern, text, f"{pattern} is not protected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
