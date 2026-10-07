#!/usr/bin/env python3
"""Generate assets/icon.png - a simple 'AI + document' glyph."""
from PIL import Image, ImageDraw

S = 512
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Rounded square background
d.rounded_rectangle([16, 16, S - 16, S - 16], radius=96, fill=(24, 39, 74, 255))

# Document sheet
d.rounded_rectangle([120, 96, 392, 416], radius=24, fill=(240, 244, 252, 255))
# Text lines
for i, y in enumerate(range(150, 280, 36)):
    d.rounded_rectangle([156, y, 356 if i % 2 == 0 else 300, y + 16], radius=8, fill=(120, 138, 176, 255))

# Sparkle / AI accent
d.polygon(
    [(360, 300), (392, 372), (464, 404), (392, 436), (360, 508), (328, 436), (256, 404), (328, 372)],
    fill=(255, 176, 32, 255),
)
d.polygon(
    [(360, 332), (380, 384), (432, 404), (380, 424), (360, 476), (340, 424), (288, 404), (340, 384)],
    fill=(255, 214, 120, 255),
)

img.save("assets/icon.png")
print("wrote assets/icon.png", img.size)
