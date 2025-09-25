from pathlib import Path
import pandas as pd

def write_fasttext(fp: Path, frame: pd.DataFrame,
                   label_col: str, text_col: str,
                   region_col: str | None = None,
                   bufsize: int = 100_000):
    fp.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        fp.write_text("", encoding="utf-8")
        return

    if region_col and region_col in frame.columns:
        regions = frame[region_col].astype(str).fillna("")
    else:
        regions = [""] * len(frame)

    with fp.open("a", encoding="utf-8") as f:
        buf = []
        for lbl, txt, region in zip(frame[label_col], frame[text_col], regions):
            buf.append(f"__label__{lbl} {txt} {region}\n")
            if len(buf) >= bufsize:
                f.writelines(buf); buf.clear()
        if buf:
            f.writelines(buf)
