from pathlib import Path

from PIL import Image


def extract(gif_path: str, out_dir: str):
    gif_path = Path(gif_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    if not gif_path.exists():
        print(f"  WARNING: {gif_path} not found, skipping.")
        return 0

    gif = Image.open(gif_path)
    n = 0
    try:
        while True:
            gif.seek(n)
            gif.convert("RGB").save(out_dir / f"frame{n}.png")
            n += 1
    except EOFError:
        pass

    print(f"  {gif_path} -> {out_dir}/ ({n} frames)")
    return n


extract("pointcloud.gif", "./")
