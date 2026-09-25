"""Generate demo_data/docs/security_policy.pdf without any PDF library.

Writes a minimal, valid PDF (Helvetica text, one text stream per page) so the demo
includes a real PDF that pypdf can extract. Run: python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "demo_data" / "docs" / "security_policy.pdf"

PAGES = [
    [
        "# Nimbus Labs Information Security Policy",
        "Version 4.1 - effective 1 July 2026. Owner: Security Team (security@nimbuslabs.example).",
        "",
        "# 1. Passwords and Authentication",
        "All accounts must use passwords of at least 14 characters. Password managers are mandatory; "
        "the company provides 1Password to every employee. Multi-factor authentication (MFA) is "
        "required for email, source control, cloud consoles and the VPN. Hardware security keys are "
        "required for administrators.",
        "",
        "# 2. API Keys and Secrets",
        "API keys and access tokens must be rotated at least every 90 days. Secrets must never be "
        "committed to source control; they are stored in the company vault. A leaked secret must be "
        "revoked immediately.",
        "",
        "# 3. Devices",
        "Company laptops must use full-disk encryption and lock automatically after 5 minutes of "
        "inactivity. Operating system security updates must be installed within 7 days of release.",
    ],
    [
        "# 4. Incident Reporting",
        "Any suspected security incident, such as a phishing email, a lost laptop or a leaked "
        "password, must be reported to the Security Team within 1 hour of discovery, by email to "
        "security@nimbuslabs.example or in the #security-incidents Slack channel.",
        "",
        "# 5. Data Classification",
        "Company data is classified into four levels: Public, Internal, Confidential and Restricted. "
        "Customer data is always classified as Restricted. Restricted data may only be stored in "
        "approved systems and must be encrypted at rest and in transit.",
        "",
        "# 6. Access Reviews",
        "Managers review the access rights of their team members every quarter. Access for employees "
        "who leave the company is removed on their last working day.",
        "",
        "# 7. Security Training",
        "All employees complete security awareness training during onboarding and then once per "
        "year. Phishing simulations are run every quarter.",
    ],
]


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(lines: list[str]) -> bytes:
    ops = ["BT", "50 790 Td"]
    for line in lines:
        if line.startswith("# "):
            ops += ["/F2 13 Tf", "18 TL", f"({_escape(line[2:])}) Tj", "T*"]
            continue
        if not line:
            ops += ["/F1 11 Tf", "8 TL", "() Tj", "T*"]
            continue
        ops += ["/F1 11 Tf", "15 TL"]
        for wrapped in textwrap.wrap(line, 88):
            ops += [f"({_escape(wrapped)}) Tj", "T*"]
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def build_pdf() -> bytes:
    objects: list[bytes] = []
    n_pages = len(PAGES)
    # 1 catalog, 2 pages, 3 font regular, 4 font bold, then (page, content) pairs
    page_ids = [5 + 2 * i for i in range(n_pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(PAGES):
        content_id = page_ids[i] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        stream = _page_stream(lines)
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build_pdf())
    print(f"wrote {OUT}")
