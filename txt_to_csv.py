# txt_to_csv_fixed_path.py
from pathlib import Path
import csv

# ==== 只改這裡即可 ====
IN_PATH  = Path("D:\\Special topic data collection(1)\\YOLO TXT/IMG_8196\\frame_00109.txt")   # 輸入 .txt
OUT_PATH = Path("frame_00109.csv")  # 輸出 .csv
DELIM = "\t"            # 你的分隔符：","、"\t"、";"、"|"
USE_WHITESPACE = False  # 若是「多個空白/空白+Tab 混用」→ 設 True
ENCODING = "utf-8"      # 需要時可改 big5、cp950 等
# =====================

def main():
    in_path  = IN_PATH.expanduser().resolve()
    out_path = OUT_PATH.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open(encoding=ENCODING, newline="") as fin, \
         out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)

        if USE_WHITESPACE:
            rows = (line.rstrip("\r\n").split() for line in fin)  # 任意長度空白切割
        else:
            rows = (line.rstrip("\r\n").split(DELIM) for line in fin)

        writer.writerows(rows)

    print(f"✅ 轉換完成 -> {out_path}")

if __name__ == "__main__":
    main()
