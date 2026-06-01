"""
Inline every image reference in a Markdown file as a base64 data URI, so the file
becomes fully self-contained (single file, no img/ folder needed).

Useful when sharing a write-up over email / chat — the recipient only needs the .md.

Input:  outputs/experiments_overview/frameworks_comaprison_results.md
Output: outputs/experiments_overview/frameworks_comparison_results_SHAREABLE.md

Usage:
    python src/learning/embed_md_images.py
    # or:
    python src/learning/embed_md_images.py path/to/source.md path/to/output.md
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SRC = PROJECT_ROOT / "outputs" / "experiments_overview" / "frameworks_comaprison_results.md"
DEFAULT_DST = PROJECT_ROOT / "outputs" / "experiments_overview" / "frameworks_comparison_results_SHAREABLE.md"

IMG_REGEX = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def embed(src_path: Path, dst_path: Path) -> None:
    src_text = src_path.read_text(encoding="utf-8")
    src_dir = src_path.parent
    replaced = 0
    skipped = 0

    def _to_data_uri(match: re.Match) -> str:
        nonlocal replaced, skipped
        alt, link = match.group(1), match.group(2)

        # Skip already-embedded data URIs and remote URLs
        if link.startswith(("data:", "http://", "https://")):
            skipped += 1
            return match.group(0)

        img_path = (src_dir / link).resolve()
        if not img_path.is_file():
            print(f"  WARNING: image not found: {img_path}")
            skipped += 1
            return match.group(0)

        suffix = img_path.suffix.lower().lstrip(".")
        mime = "image/png" if suffix == "png" else f"image/{suffix}"
        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        replaced += 1
        return f"![{alt}](data:{mime};base64,{b64})"

    new_text = IMG_REGEX.sub(_to_data_uri, src_text)

    dst_path.write_text(new_text, encoding="utf-8")

    src_kb = src_path.stat().st_size / 1024
    dst_kb = dst_path.stat().st_size / 1024
    print(f"Source:      {src_path}")
    print(f"Output:      {dst_path}")
    print(f"  Images embedded: {replaced}")
    print(f"  Images skipped:  {skipped}")
    print(f"  Size: {src_kb:.1f} KB -> {dst_kb:.1f} KB")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DST
    if not src.exists():
        print(f"Source file not found: {src}", file=sys.stderr)
        sys.exit(1)
    embed(src, dst)


if __name__ == "__main__":
    main()
