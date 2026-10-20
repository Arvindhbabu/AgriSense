import unittest
from PIL import Image

from utils.image_quality import assess_image_quality


class ImageQualityTests(unittest.TestCase):
    def test_dark_uniform_input_requires_review(self):
        image = Image.new("RGB", (640, 480), color=(0, 0, 0))
        result = assess_image_quality(image)
        self.assertTrue(result["review_required"])
        self.assertIn("severely_underexposed", result["warnings"])
        self.assertIn("very_low_contrast_or_nearly_uniform_image", result["warnings"])
        self.assertFalse(result["score_is_calibrated"])

    def test_reasonably_exposed_detailed_input_passes_acquisition_gate(self):
        image = Image.new("RGB", (640, 480))
        pixels = image.load()
        for y in range(480):
            for x in range(640):
                value = 70 + ((x * 7 + y * 11) % 150)
                pixels[x, y] = (value, min(255, value + 10), max(0, value - 10))
        result = assess_image_quality(image)
        self.assertFalse(result["review_required"])
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["height"], 480)
        self.assertGreater(result["quality_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
