"""Export helpers: Markdown (.md) and self-contained HTML (.html)."""

from __future__ import annotations

import base64
import mimetypes
import re
import shutil
from pathlib import Path
from typing import List

_IMG_SRC_RE = re.compile(r'src="(images/[^"]+)"')

HTML_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; color: #1a1a1a; }
h1, h2, h3 { line-height: 1.25; }
table { border-collapse: collapse; margin: 1rem 0; }
th, td { border: 1px solid #c9c9c9; padding: 6px 10px; }
img { max-width: 100%; }
code, pre { background: #f4f4f4; border-radius: 4px; }
pre { padding: 0.8rem; overflow-x: auto; }
hr { border: 0; border-top: 1px solid #ddd; margin: 2rem 0; }
blockquote { color: #555; border-left: 4px solid #ddd; margin-left: 0; padding-left: 1rem; }
"""


def page_separator() -> str:
    return "\n\n---\n\n"


def assemble_markdown(pages: List[str]) -> str:
    return page_separator().join(p.strip() for p in pages if p and p.strip())


def export_markdown(md_text: str, images_dir: Path, dest: Path) -> Path:
    """Write the .md file and copy the cropped page images next to it."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md_text, encoding="utf-8")
    if images_dir and Path(images_dir).is_dir():
        target = dest.parent / "images"
        src = Path(images_dir).resolve()
        if src != Path(target).resolve() and src not in Path(target).resolve().parents:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
    return dest


def _to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def markdown_to_html(md_text: str, base_dir: Path, embed_images: bool = True) -> str:
    """Convert markdown to a full HTML document.

    When `embed_images` is set, relative `images/...` sources are inlined as
    base64 data URIs so the .html file is fully self-contained.
    """
    import markdown as _markdown

    body = _markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists"]
    )

    if embed_images and base_dir and base_dir.is_dir():
        def _repl(match: re.Match) -> str:
            rel = match.group(1)
            candidate = (base_dir / rel).resolve()
            if candidate.is_file():
                return f'src="{_to_data_uri(candidate)}"'
            return match.group(0)

        body = _IMG_SRC_RE.sub(_repl, body)

    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>AiOCR export</title>\n<style>{HTML_CSS}</style>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


def export_html(md_text: str, base_dir: Path, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown_to_html(md_text, base_dir, embed_images=True), encoding="utf-8")
    return dest
