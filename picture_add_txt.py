# add_top_right_text.py
# pip install --upgrade pillow
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Optional, Tuple

ColorRGB = Tuple[int, int, int]
ColorRGBA = Tuple[int, int, int, int]

def add_text_top_right(
    image_path: str,
    text: str,
    out_path: Optional[str] = None,
    pad_right: int = 60,                        # ★ 與右邊距離（越大越往內）
    pad_top: int = 40,                          # ★ 與上邊距離
    font_size: Optional[int] = None,
    font_path: Optional[str] = None,
    # ---- 外觀（預設：白底黑字）----
    text_fill: ColorRGBA = (0, 0, 0, 255),
    bg_color: ColorRGB = (255, 255, 255),
    bg_alpha: int = 255,
    corner_radius: int = 10,
    # 內距（白框與文字距離）
    text_pad_x: int = 12,
    text_pad_y: int = 10,
    # 陰影/描邊
    shadow: bool = True,
    shadow_offset: Tuple[int, int] = (1, 1),
    shadow_fill: ColorRGBA = (0, 0, 0, 160),
    outline_width: int = 0,
    outline_fill: ColorRGBA = (0, 0, 0, 220),
):
    img = Image.open(image_path).convert("RGBA")

    if font_size is None:
        font_size = max(24, img.width // 30)

    # 載入字型
    font = None
    try_paths = [font_path] if font_path else []
    try_paths += [
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhl.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in try_paths:
        try:
            if p and Path(p).exists():
                font = ImageFont.truetype(p, font_size)
                break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 先以 (0,0) 估字寬高，用於反推出 x
    try:
        tmp_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=outline_width)
        text_w = tmp_bbox[2] - tmp_bbox[0]
        text_h = tmp_bbox[3] - tmp_bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)

    # 右上角定位（距離右邊 pad_right、距離上邊 pad_top）
    x = img.width - pad_right - text_w
    y = pad_top

    # 以實際座標再次量測（包含描邊）
    try:
        left, top, right, bottom = draw.textbbox(
            (x, y), text, font=font, stroke_width=outline_width
        )
    except AttributeError:
        left, top = x, y
        right, bottom = x + text_w, y + text_h

    # 白框：內距 + 陰影外擴
    pad_l = text_pad_x
    pad_t = text_pad_y
    pad_r = text_pad_x + (shadow_offset[0] if shadow else 0)
    pad_b = text_pad_y + (shadow_offset[1] if shadow else 0)
    rect = [left - pad_l, top - pad_t, right + pad_r, bottom + pad_b]

    if bg_alpha > 0:
        rgba = (bg_color[0], bg_color[1], bg_color[2], int(bg_alpha))
        try:
            r = min(corner_radius, int((rect[3] - rect[1]) / 2))
            draw.rounded_rectangle(rect, radius=r, fill=rgba)
        except AttributeError:
            draw.rectangle(rect, fill=rgba)

    # 陰影
    if shadow:
        draw.text((x + shadow_offset[0], y + shadow_offset[1]),
                  text, font=font, fill=shadow_fill)

    # 描邊
    if outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline_fill)

    # 主要文字
    draw.text((x, y), text, font=font, fill=text_fill)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    if not out_path:
        p = Path(image_path)
        out_path = str(p.with_name(p.stem + "_with_text" + p.suffix))
    out.save(out_path, quality=95)
    return out_path


if __name__ == "__main__":
    # === 依你的檔案調整 ===
    src = "frame_00109_angle.jpg"
    dst = "output_frame_00109_angle_txt.jpg"
    fontfile = r"C:\Windows\Fonts\msjh.ttc"

    # 白底（不透明）＋黑字，並調整離邊界距離
    print(add_text_top_right(
        src, "基礎高壓發球", out_path=dst, font_path=fontfile,
        font_size=128,
        bg_color=(255, 255, 255), bg_alpha=255,
        text_fill=(0, 0, 0, 255),
        corner_radius=14,
        text_pad_x=12, text_pad_y=10,
        shadow=True, shadow_offset=(1, 1),
        outline_width=0,
        pad_right=80,      # ← 跟右邊保持 80px
        pad_top=60         # ← 跟上邊保持 60px
    ))
