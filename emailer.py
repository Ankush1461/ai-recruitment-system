# ================================================================
# 📧 Email Sender — shortlist notifications & interview invites
# ================================================================
# Free: every account brings its OWN SMTP server (e.g. Gmail app password,
# Outlook, or any free provider) saved from the Email tab → ⚙️ Email settings.
# The shared .env values are intentionally NOT used — an account that hasn't
# configured its own sender gets a clear "not configured" message.

from __future__ import annotations

import html
import re
import smtplib
import ssl
from email.message import EmailMessage

import db

_DEFAULT_FROM_NAME = "TalentIQ Recruiter"


def resolved_settings() -> dict:
    """The SMTP settings in effect — ONLY the signed-in account's own config
    (saved from Email tab → ⚙️ Email settings). No .env fallback: an account
    that hasn't saved a sender is simply not configured."""
    cfg = db.get_email_settings() or {}
    return {
        "host": (cfg.get("host") or "").strip(),
        "port": int(cfg.get("port") or 587),
        "mail_from": (cfg.get("mail_from") or "").strip(),
        "mail_from_name": (cfg.get("mail_from_name") or "").strip()
        or _DEFAULT_FROM_NAME,
        "user": (cfg.get("user") or "").strip(),
        "password": cfg.get("password") or "",
        "starttls": str(cfg.get("starttls", "1")) != "0",
        # True when an encrypted stored password no longer decrypts (the
        # account password changed → different key) — the UI warns the
        # recruiter to re-enter the SMTP password.
        "password_unreadable": bool(cfg.get("password_unreadable")),
    }


def is_configured() -> bool:
    """True when the account's own SMTP sender is fully configured."""
    s = resolved_settings()
    return bool(s["host"] and s["mail_from"])


def extract_email(text: str | None) -> str | None:
    """Pull the first email address out of a text blob (e.g. a resume)."""
    if not text:
        return None
    m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+[.][a-zA-Z]{2,}", text)
    return m.group(0) if m else None


# ---- Templates ---------------------------------------------------------------

def _html_page(inner: str) -> str:
    return (
        '<div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;'
        'margin:auto;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden">'
        '<div style="background:#0f766e;color:#ffffff;padding:16px 24px;'
        'font-size:18px;font-weight:700">TalentIQ · AI Recruiter</div>'
        f'<div style="padding:24px;color:#0f172a">{inner}</div></div>'
    )


def _p(text: str) -> str:
    """Wrap already-escaped / HTML-safe content in a paragraph tag.

    Callers are responsible for `html.escape()`-ing every dynamic value
    before passing it in; the static template markup (`<strong>`, `<br/>`)
    must pass through RAW so it renders as HTML instead of literal text.
    """
    return f"<p>{text}</p>"


def build_shortlist_email(
    job: dict,
    candidate: dict,
    extra_msg: str = "",
) -> tuple[str, str]:
    """(subject, html_body) for a shortlist notification."""
    name = html.escape(candidate.get("name") or "there")
    title = html.escape(job.get("title", ""))
    req = html.escape(job.get("req_id") or "")
    subject = f"Shortlisted: {job.get('title', '')} ({req or 'application'})"
    body = _html_page(
        _p(f"Hi {name},")
        + _p(
            f"Great news — your application for <strong>{title}</strong> "
            f"({req}) has been <strong>shortlisted</strong> by our AI screening."
        )
        + _p(
            "Your background is a strong match for the role's requirements, "
            "and you are now in the candidate pipeline."
        )
        + (f"<p style='padding:12px;background:#f0fdfa;border-left:3px solid #0f766e'>{html.escape(extra_msg)}</p>" if extra_msg else "")
        + _p("Next step: the recruiting team will reach out to schedule an interview.")
        + _p("Best regards,<br/>The TalentIQ Recruiting Team")
    )
    return subject, body


def build_invite_email(
    job: dict,
    candidate: dict,
    questions: list[str] | None = None,
    extra_msg: str = "",
    invite_link: str = "",
) -> tuple[str, str]:
    """(subject, html_body) for an interview invitation (with sample questions).

    `invite_link` is optional — when provided it is rendered as a prominent
    "Join the interview" button in the email body.
    """
    name = html.escape(candidate.get("name") or "there")
    title = html.escape(job.get("title", ""))
    req = html.escape(job.get("req_id") or "")
    subject = f"Interview invitation: {job.get('title', '')} ({req or 'application'})"
    inner = (
        _p(f"Hi {name},")
        + _p(
            f"Congratulations — you have been invited to a technical interview "
            f"for <strong>{title}</strong> ({req})."
        )
        + _p("The interview covers your background and the role's key requirements.")
    )
    if invite_link:
        link = invite_link.strip()
        # Only http(s) links become clickable buttons — anything else (e.g. a
        # javascript: scheme) degrades to plain text so it can never be abused.
        href = html.escape(link)
        if link.lower().startswith(("http://", "https://")):
            inner += (
                "<p>Join your interview using the link below:</p>"
                '<p style="text-align:center">'
                f'<a href="{href}" style="display:inline-block;padding:11px 20px;'
                "background:#0f766e;color:#ffffff;text-decoration:none;"
                'border-radius:8px;font-weight:600">Join the interview</a></p>'
            )
        # Always include the raw URL as visible text too, so plain-text email
        # clients (which strip HTML) still see where to join.
        inner += _p(f"Interview link: {link}")
    if questions:
        items = "".join(f"<li>{html.escape(q)}</li>" for q in questions[:5])
        inner += f"<p>Expect questions such as:</p><ul>{items}</ul>"
    inner += (
        (f"<p style='padding:12px;background:#f0fdfa;border-left:3px solid #0f766e'>{html.escape(extra_msg)}</p>" if extra_msg else "")
        + _p("Please reply to confirm a convenient time.")
        + _p("Best regards,<br/>The TalentIQ Recruiting Team")
    )
    return subject, _html_page(inner)


def _text_version(html_body: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html_body).split())


def send_email(to: str, subject: str, body_html: str) -> dict:
    """Send one email via SMTP. Returns {'ok': bool, 'error': str | None, ...}.

    On success the send is recorded in the audit log (action='email_sent').
    """
    to = (to or "").strip()
    if not to or "@" not in to:
        return {"ok": False, "error": "No valid recipient email address."}
    if not is_configured():
        return {
            "ok": False,
            "error": (
                "SMTP is not configured. Open **Email tab → ⚙️ Email settings** "
                "and save your own SMTP sender (host + from-address) — e.g. a "
                "Gmail app password. Every account configures its own."
            ),
        }
    s = resolved_settings()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{s['mail_from_name']} <{s['mail_from']}>"
    msg["To"] = to
    msg.set_content(_text_version(body_html))
    msg.add_alternative(body_html, subtype="html")
    try:
        with smtplib.SMTP(s["host"], int(s["port"] or 587), timeout=25) as conn:
            if s["starttls"]:
                conn.starttls(context=ssl.create_default_context())
            if s["user"]:
                conn.login(s["user"], s["password"])
            conn.send_message(msg)
    except Exception as e:
        return {"ok": False, "error": f"SMTP send failed: {e}"}
    db.audit("email_sent", "email", to, subject)
    return {"ok": True, "error": None, "subject": subject, "to": to}
