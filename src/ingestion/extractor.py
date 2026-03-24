"""
Universal Document Extractor — converts any resume format to clean text.

Format detection is MIME-first (via `file --mime-type`), with extension as
fallback only. This handles renamed files, extensionless files, and formats
inside ZIPs correctly.

Supported: PDF, DOCX, DOC, RTF, HTML, plain text, LaTeX, images (OCR).
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

# Extension map kept only as a last-resort fallback — MIME detection is primary.
# Not used to gate which files are processed; use try-extract pattern instead.
EXTENSION_MAP = {
    '.txt': 'text', '.md': 'text', '.tex': 'latex', '.csv': 'text',
    '.pdf': 'pdf',
    '.docx': 'docx', '.doc': 'doc_legacy',
    '.rtf': 'rtf',
    '.html': 'html', '.htm': 'html',
    '.png': 'image', '.jpg': 'image', '.jpeg': 'image',
    '.webp': 'image', '.tiff': 'image', '.tif': 'image',
}

# MIME → extractor key
MIME_MAP = [
    ('application/pdf',                    'pdf'),
    ('application/msword',                 'doc_legacy'),
    ('application/vnd.openxmlformats',     'docx'),
    ('application/vnd.ms-word',            'doc_legacy'),
    ('image/',                             'image'),
    ('text/html',                          'html'),
    ('text/rtf',                           'rtf'),
    ('application/rtf',                    'rtf'),
    ('text/',                              'text'),   # text/plain, text/x-tex, etc.
]


def detect_format(filepath: str) -> str:
    """Detect document format. MIME type is primary; extension is fallback."""
    ext = Path(filepath).suffix.lower()

    # 1. MIME via system `file` command (works on extensionless / renamed files)
    try:
        result = subprocess.run(
            ['file', '--brief', '--mime-type', filepath],
            capture_output=True, text=True, timeout=5,
        )
        mime = result.stdout.strip().lower()
        for prefix, fmt in MIME_MAP:
            if mime.startswith(prefix):
                # .tex files report text/plain or text/x-tex — override to latex
                # so raw LaTeX commands are stripped before scoring.
                if fmt == 'text' and ext == '.tex':
                    return 'latex'
                return fmt
    except Exception:
        pass

    # 2. Extension fallback
    return EXTENSION_MAP.get(ext, 'unknown')


def extract_text(filepath: str) -> str:
    """Extract plain text from any supported document. Returns '' on failure."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fmt = detect_format(filepath)
    extractors = {
        'text':       lambda f: open(f, 'r', encoding='utf-8', errors='replace').read(),
        'latex':      extract_latex,
        'pdf':        extract_pdf,
        'docx':       extract_docx,
        'doc_legacy': extract_doc_legacy,
        'rtf':        extract_rtf,
        'html':       extract_html,
        'image':      extract_image,
    }

    # Re-check for LaTeX even if MIME said 'text' and extension override didn't fire
    # (e.g. extensionless .tex read from a ZIP)
    if fmt == 'text':
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as _f:
                _head = _f.read(512)
            if re.search(r'\\documentclass|\\begin\{document\}|\\usepackage', _head):
                fmt = 'latex'
        except Exception:
            pass

    extractor = extractors.get(fmt)
    if not extractor:
        return ""

    text = extractor(filepath)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def ocr_pdf(filepath: str, reader=None) -> str:
    """OCR fallback for scanned / image-only PDFs.

    Strategy (in order):
      1. Pull embedded images out of each PDF page via pypdf and OCR them.
      2. If no images found that way, render each page to a temp PNG via
         the `convert` CLI (ImageMagick) and OCR the PNG.
    Returns '' if all strategies fail or tesseract is not installed.
    """
    import io

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    pages_text = []

    # Strategy 1: pypdf embedded image extraction
    try:
        from pypdf import PdfReader
        if reader is None:
            reader = PdfReader(filepath)
        for page in reader.pages:
            page_parts = []
            for img_obj in page.images:
                try:
                    pil_img = Image.open(io.BytesIO(img_obj.data))
                    page_parts.append(pytesseract.image_to_string(pil_img))
                except Exception:
                    continue
            if page_parts:
                pages_text.append("\n".join(page_parts))
    except Exception:
        pass

    if pages_text and any(t.strip() for t in pages_text):
        return "\n\n".join(pages_text)

    # Strategy 2: render PDF pages to images via ImageMagick `convert`
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pattern = os.path.join(tmpdir, "page.png")
            result = subprocess.run(
                ['convert', '-density', '200', filepath, out_pattern],
                capture_output=True, timeout=60,
            )
            png_files = sorted(Path(tmpdir).glob("page*.png"))
            for png in png_files:
                try:
                    pil_img = Image.open(str(png))
                    pages_text.append(pytesseract.image_to_string(pil_img))
                except Exception:
                    continue
    except Exception:
        pass

    return "\n\n".join(pages_text) if pages_text else ""


def extract_pdf(filepath: str) -> str:
    # Attempt 1: pypdf text layer
    reader = None
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        if len(text.strip()) > 100:
            return text
    except Exception:
        pass

    # Attempt 2: pdftotext CLI
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', filepath, '-'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 100:
            return result.stdout
    except Exception:
        pass

    # Attempt 3: OCR — scanned / image-only PDF.
    # Extract embedded images via pypdf and run tesseract on each page image.
    return ocr_pdf(filepath, reader)


def extract_latex(filepath: str) -> str:
    """Convert a LaTeX source file to clean plain text.

    Tries pandoc first (best quality), then falls back to a regex stripper
    that removes commands while preserving their text arguments.
    """
    # Attempt 1: pandoc — handles virtually all LaTeX resume templates cleanly.
    try:
        result = subprocess.run(
            ['pandoc', filepath, '-t', 'plain', '--wrap=none'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 50:
            return result.stdout
    except Exception:
        pass

    # Attempt 2: regex stripper — good enough for structured resume LaTeX.
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            src = fh.read()

        # Remove LaTeX comments
        src = re.sub(r'%.*', '', src)
        # Remove preamble (everything before \begin{document})
        src = re.sub(r'(?s).*?\\begin\{document\}', '', src, count=1)
        # Remove \end{document} and anything after
        src = re.sub(r'\\end\{document\}.*', '', src, flags=re.DOTALL)
        # Unwrap common formatting commands, keeping their argument: \textbf{X} → X
        for cmd in ('textbf', 'textit', 'emph', 'underline', 'texttt',
                    'section', 'subsection', 'subsubsection',
                    'item', 'textsc', 'textrm', 'textsf'):
            src = re.sub(rf'\\{cmd}\s*\{{([^}}]*)\}}', r'\1', src)
        # Remove href but keep display text: \href{url}{text} → text
        src = re.sub(r'\\href\{[^}]*\}\{([^}]*)\}', r'\1', src)
        # Remove environments wrapper tags but keep content
        src = re.sub(r'\\(?:begin|end)\{[^}]+\}', '', src)
        # Remove remaining commands with arguments
        src = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', src)
        # Remove bare commands (no arguments)
        src = re.sub(r'\\[a-zA-Z]+\*?', '', src)
        # Remove leftover braces
        src = re.sub(r'[{}]', '', src)
        # Clean up whitespace
        src = re.sub(r'[ \t]+', ' ', src)
        src = re.sub(r'\n{3,}', '\n\n', src)
        return src.strip()
    except Exception:
        return ""


def extract_docx(filepath: str) -> str:
    try:
        result = subprocess.run(
            ['pandoc', filepath, '-t', 'plain', '--wrap=none'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        pass
    try:
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        pass
    return ""


def extract_doc_legacy(filepath: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run(
                ['libreoffice', '--headless', '--convert-to', 'docx', '--outdir', tmpdir, filepath],
                capture_output=True, timeout=60, check=True,
            )
            converted = list(Path(tmpdir).glob('*.docx'))
            if converted:
                return extract_docx(str(converted[0]))
        except Exception:
            pass
    return ""


def extract_rtf(filepath: str) -> str:
    try:
        result = subprocess.run(
            ['pandoc', filepath, '-t', 'plain', '--wrap=none'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return open(filepath, 'r', encoding='utf-8', errors='replace').read()


def extract_html(filepath: str) -> str:
    try:
        from bs4 import BeautifulSoup
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        return soup.get_text(separator='\n')
    except ImportError:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            html = f.read()
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        return re.sub(r'<[^>]+>', ' ', text)


def extract_image(filepath: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(filepath))
    except Exception:
        return ""


# Files to skip regardless of MIME (system / hidden / archive files)
SKIP_NAMES = {'.DS_Store', 'Thumbs.db', '__MACOSX'}
SKIP_EXTS = {'.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.exe', '.dmg'}


def extract_directory(dir_path: str) -> Dict[str, str]:
    """
    Extract text from all files in a directory.
    Uses MIME detection — no extension allowlist. Skips files that return empty text.
    """
    results = {}
    p = Path(dir_path)
    if not p.exists():
        return results

    for f in sorted(p.iterdir()):
        if not f.is_file():
            continue
        if f.name in SKIP_NAMES or f.name.startswith('.') or f.name.startswith('__'):
            continue
        if f.suffix.lower() in SKIP_EXTS:
            continue
        try:
            text = extract_text(str(f))
            if text and len(text.strip()) > 50:
                results[f.stem] = text
        except Exception:
            pass

    return results
