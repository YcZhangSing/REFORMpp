#!/usr/bin/env python3
"""Build the 40 annotated cross-split pairs used by the project page.

A pair is included only when the CSV label cell is non-empty. The script
stops if that filter does not yield exactly 40 pairs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path("/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/REFORMpp")
CSV_PATH = Path(
    "/home/star/.cursor/projects/mnt-da36552c-a636-46f9-9a37-676e692003a2-yuchen/attachments/6acf260b-6590-4621-9c1c-5be3ad66de54/audit_annotations.csv"
)
MANIFEST = Path(
    "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/PAMI_paper_scripts/emb_sim/manifest/rom_manifest.sqlite"
)
CANDIDATES = Path(
    "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/PAMI_paper_scripts/emb_sim/retrieval_audit/candidates"
)
OUT_DIR = REPO / "assets" / "cross_split_audit"
IMAGE_DIR = OUT_DIR / "images"
JSON_PATH = OUT_DIR / "cross_split_demos.json"

LABELS = {
    "orig": "Authentic",
    "face_swap": "Face Swap",
    "face_attribute": "Face Attribute Editing",
    "bg_rep": "Background Replacement",
    "full_gene": "Full-image Generation",
    "text_swap": "Text Replacement",
}
SPLIT_NAME = {"train": "Train", "val": "Val", "test": "Test"}
RETRIEVAL_NAME = {
    "image": "Image Retrieval",
    "caption": "Caption Retrieval",
    "joint": "Image–Caption Retrieval",
}
MODE_ORDER = ("image", "caption", "joint")
DISPLAY_FIRST = ("ROMNYT00004020", "ROMBBC00058637")


def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def flag(value) -> bool:
    return clean(value).lower() in {"1", "1.0", "true", "yes"}


def manipulation_name(fake_cls: str) -> str:
    parts = str(fake_cls).split("&")
    unknown = [part for part in parts if part not in LABELS]
    if unknown:
        raise SystemExit(f"unmapped fake_cls: {fake_cls}")
    return " + ".join(LABELS[part] for part in parts)


def load_pool() -> pd.DataFrame:
    frames = []
    for path in CANDIDATES.glob("*_*_top50.csv"):
        if path.name.startswith(("nyt_", "guardian_")) or "global" in path.name:
            continue
        frames.append(pd.read_csv(path))
    pool = pd.concat(frames, ignore_index=True)
    pool["key"] = (
        pool["embedding_mode"].astype(str)
        + "|"
        + pool["query_row_id"].astype(int).astype(str)
        + "|"
        + pool["reference_row_id"].astype(int).astype(str)
    )
    return pool.drop_duplicates("key")


def sample_record(con: sqlite3.Connection, sample_id: str) -> dict:
    row = con.execute(
        "SELECT sample_id, split, domain, fake_cls, caption, image_path FROM samples WHERE sample_id=?",
        (sample_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"sample missing from manifest: {sample_id}")
    if not Path(row["image_path"]).is_file():
        raise SystemExit(f"image missing: {row['image_path']}")
    return {
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "split": SPLIT_NAME[row["split"]],
        "manipulation_type": manipulation_name(row["fake_cls"]),
        "caption": row["caption"],
        "image_path": row["image_path"],
    }


def save_image(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        image.save(destination, "JPEG", quality=88, optimize=True)


def interleave(groups: dict[str, list[dict]]) -> list[dict]:
    cursors = {mode: 0 for mode in MODE_ORDER}
    ordered = []
    while len(ordered) < sum(len(items) for items in groups.values()):
        progressed = False
        for mode in MODE_ORDER:
            index = cursors[mode]
            if index < len(groups[mode]):
                ordered.append(groups[mode][index])
                cursors[mode] += 1
                progressed = True
        if not progressed:
            break
    return ordered


def apply_display_order(ordered: list[dict]) -> list[dict]:
    """Keep the interleaved order, then place one requested pair first.

    The pair that previously occupied the first position moves to the end.
    """
    def ident(item: dict) -> tuple[str, str]:
        return (item["reference"]["sample_id"], item["query"]["sample_id"])

    matches = [index for index, item in enumerate(ordered) if ident(item) == DISPLAY_FIRST]
    if len(matches) != 1:
        raise SystemExit(f"expected one pair {DISPLAY_FIRST}, found {len(matches)}")
    target = ordered.pop(matches[0])
    former_first = ordered.pop(0)
    ordered.insert(0, target)
    ordered.append(former_first)
    return ordered


def main() -> int:
    frame = pd.read_csv(CSV_PATH)
    required = {
        "pair_key",
        "embedding_mode",
        "similarity",
        "label",
        "near_duplicate_image",
        "paraphrased_caption",
        "same_identity",
        "note",
        "query_sample_id",
        "reference_sample_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")
    frame["label_clean"] = frame["label"].map(clean)
    labeled = frame[frame["label_clean"] != ""].copy()
    if len(labeled) != 40:
        print("Refusing to build the page.")
        print(f"Filter: non-empty label after stripping whitespace.")
        print(f"CSV rows: {len(frame)}")
        print(f"Labeled rows: {len(labeled)}")
        print(labeled["label_clean"].value_counts().to_string() if len(labeled) else "no labels")
        raise SystemExit(1)

    pool = load_pool()
    con = sqlite3.connect(MANIFEST)
    con.row_factory = sqlite3.Row
    groups = {mode: [] for mode in MODE_ORDER}
    for record in labeled.to_dict(orient="records"):
        mode = clean(record["embedding_mode"])
        if mode not in RETRIEVAL_NAME:
            raise SystemExit(f"unknown retrieval type: {mode}")
        key = clean(record["pair_key"]).replace('"', "")
        meta = pool[pool["key"] == key]
        if meta.empty:
            raise SystemExit(f"pair not found in candidate tables: {key}")
        meta_row = meta.iloc[0]
        reference = sample_record(con, clean(record["reference_sample_id"]))
        query = sample_record(con, clean(record["query_sample_id"]))
        if reference["domain"] != clean(meta_row["reference_domain"]) or query["domain"] != clean(meta_row["query_domain"]):
            raise SystemExit(f"domain mismatch for {key}")
        groups[mode].append(
            {
                "retrieval_type": mode,
                "retrieval_label": RETRIEVAL_NAME[mode],
                "similarity": round(float(record["similarity"]), 2),
                "similarity_raw": float(record["similarity"]),
                "reference": reference,
                "query": query,
                "primary_annotation": record["label_clean"],
                "near_duplicate_image": flag(record["near_duplicate_image"]),
                "paraphrased_caption": flag(record["paraphrased_caption"]),
                "same_identity": flag(record["same_identity"]),
                "note": clean(record["note"]),
            }
        )
    con.close()
    for mode in MODE_ORDER:
        groups[mode].sort(key=lambda item: (-item["similarity_raw"], item["reference"]["sample_id"], item["query"]["sample_id"]))
    ordered = apply_display_order(interleave(groups))
    if len(ordered) != 40:
        raise SystemExit(f"interleave produced {len(ordered)} pairs")
    first_modes = {item["retrieval_type"] for item in ordered[:8]}
    if first_modes != set(MODE_ORDER):
        raise SystemExit(f"first 8 do not cover all retrieval types: {sorted(first_modes)}")

    if IMAGE_DIR.exists():
        for old in IMAGE_DIR.glob("*"):
            old.unlink()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    for index, item in enumerate(ordered, start=1):
        copied = {}
        for role in ("reference", "query"):
            sample = item[role]
            filename = f"{item['retrieval_type']}_{index:02d}_{sample['sample_id']}.jpg"
            save_image(sample["image_path"], IMAGE_DIR / filename)
            copied[role] = {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "split": sample["split"],
                "manipulation_type": sample["manipulation_type"],
                "caption": sample["caption"],
                "image": f"assets/cross_split_audit/images/{filename}",
            }
        payload.append(
            {
                "pair_index": index,
                "retrieval_type": item["retrieval_type"],
                "retrieval_label": item["retrieval_label"],
                "similarity": item["similarity"],
                "similarity_raw": item["similarity_raw"],
                "reference": copied["reference"],
                "query": copied["query"],
                "primary_annotation": item["primary_annotation"],
                "auxiliary_flags": {
                    "near_duplicate_image": item["near_duplicate_image"],
                    "paraphrased_caption": item["paraphrased_caption"],
                    "same_identity": item["same_identity"],
                },
                "note": item["note"],
            }
        )
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    counts = {mode: sum(item["retrieval_type"] == mode for item in payload) for mode in MODE_ORDER}
    print(json.dumps({"pairs": len(payload), "counts": counts, "images": len(list(IMAGE_DIR.glob('*.jpg')))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
