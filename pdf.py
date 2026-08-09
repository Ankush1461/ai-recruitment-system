# ================================================================
# 📄 PDF Text Extraction
# ================================================================

from pypdf import PdfReader


def extract_pdf_text(file_obj) -> str:
    """Extract complete text from a PDF resume.

    Accepts a Gradio file object (has .name attribute), a file path string,
    or None. Returns empty string on failure.
    """
    if file_obj is None:
        return ""

    # Resolve path from Gradio file object or raw string
    file_path: str | None = getattr(file_obj, "name", None) or (
        file_obj if isinstance(file_obj, str) else None
    )
    if not file_path:
        return ""

    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception:
        pass
    return text.strip()
