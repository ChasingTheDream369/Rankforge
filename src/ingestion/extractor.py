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
    '.txt': 'text', '.md': 'text', '.tex': 'text', '.csv': 'text',
    '.pdf': 'pdf',
    '.docx': 'docx', '.doc': 'doc_legacy',
    '.rtf': 'rtf',
    '.html': 'html', '.htm': 'html',
    '.png': 'image', '.jpg': 'image', '.jpeg': 'image',
    '.webp': 'image', '.tiff': 'image', '.tif': 'image',
}

# MIME → extractor key
_MIME_MAP = [
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
    # 1. MIME via system `file` command (works on extensionless / renamed files)
    try:
        result = subprocess.run(
            ['file', '--brief', '--mime-type', filepath],
            capture_output=True, text=True, timeout=5,
        )
        mime = result.stdout.strip().lower()
        for prefix, fmt in _MIME_MAP:
            if mime.startswith(prefix):
                return fmt
    except Exception:
        pass

    # 2. Extension fallback
    ext = Path(filepath).suffix.lower()
    return EXTENSION_MAP.get(ext, 'unknown')


def extract_text(filepath: str) -> str:
    """Extract plain text from any supported document. Returns '' on failure."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fmt = detect_format(filepath)
    extractors = {
        'text':       lambda f: open(f, 'r', encoding='utf-8', errors='replace').read(),
        'pdf':        extract_pdf,
        'docx':       extract_docx,
        'doc_legacy': extract_doc_legacy,
        'rtf':        extract_rtf,
        'html':       extract_html,
        'image':      extract_image,
    }

    extractor = extractors.get(fmt)
    if not extractor:
        return ""

    text = extractor(filepath)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_pdf(filepath: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        if len(text.strip()) > 100:
            return text
    except Exception:
        pass
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', filepath, '-'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 100:
            return result.stdout
    except Exception:
        pass
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
_SKIP_NAMES = {'.DS_Store', 'Thumbs.db', '__MACOSX'}
_SKIP_EXTS  = {'.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.exe', '.dmg'}


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
        if f.name in _SKIP_NAMES or f.name.startswith('.') or f.name.startswith('__'):
            continue
        if f.suffix.lower() in _SKIP_EXTS:
            continue
        try:
            text = extract_text(str(f))
            if text and len(text.strip()) > 50:
                results[f.stem] = text
        except Exception:
            pass

    return results
