"""1280x400 black-and-white banner: pixel blackletter title, mono tagline, dithered Durer plates."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ART = os.path.join(os.path.dirname(__file__), "..", "docs", "art")
W, H = 1280, 400


def plate(name, width, flip=False, top_frac=0.5):
    im = Image.open(os.path.join(ART, name)).convert("L")
    mx, my = int(im.width * 0.04), int(im.height * 0.04)
    im = im.crop((mx, my, im.width - mx, im.height - my))  # drop the plate border
    scale = max(width / im.width, H / im.height)
    im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1), Image.LANCZOS)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    top = int(max(0, im.height - H) * top_frac)
    left = (im.width - width) // 2
    im = im.crop((left, top, left + width, top + H))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(im, dtype=float) / 255
    lo, hi = np.percentile(a, [5, 97])
    return np.clip((a - lo) / (hi - lo), 0, 1) ** 2.2  # paper stays light, hatching goes dark


def fade(a, side):
    x = np.linspace(0, 1, a.shape[1])
    ramp = np.clip((1 - x) / 0.85, 0, 1) if side == "left" else np.clip(x / 0.85, 0, 1)
    return a * ramp[None, :] ** 1.6


def pixel_text(text, font, size, scale):
    f = ImageFont.truetype(font, size)
    box = f.getbbox(text)
    im = Image.new("1", (box[2] - box[0] + 4, box[3] - box[1] + 4), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    d.text((2 - box[0], 2 - box[1]), text, font=f, fill=1)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main():
    canvas = np.zeros((H, W))
    canvas[:, :360] = np.maximum(canvas[:, :360], fade(plate("durer_knight_death_devil.jpg", 360), "left"))
    canvas[:, W - 360:] = np.maximum(canvas[:, W - 360:], fade(plate("durer_four_riders.jpg", 360, flip=True), "right"))
    img = Image.fromarray((canvas * 255).astype("uint8")).convert("1")  # Floyd-Steinberg to two tones
    img = img.convert("L").point(lambda v: 200 if v else 0)
    title = pixel_text("harnesslag", os.path.join(ART, "UnifrakturMaguntia-Book.ttf"), 34, 4)
    img.paste(255, ((W - title.width) // 2, 70), title)
    y = 70 + title.height + 26
    for line in ["re-render stale agent rollouts,", "split the log-ratio, keep learning"]:
        t = pixel_text(line, "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13, 2)
        img.paste(170, ((W - t.width) // 2, y), t)
        y += t.height + 8
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "screenshots", "banner.png")
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
