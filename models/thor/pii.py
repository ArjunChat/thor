"""Personal data (PII) detection and replacement.

Detects email addresses, phone numbers, IP addresses, payment card numbers
(Luhn-checked), IBANs (mod-97-checked) and US social security numbers.

Two modes:
- "surrogate" (for training data): replace each value with a fictitious but
  realistic one from reserved ranges (example.com emails, 555-01xx phone numbers,
  RFC 5737 documentation IPs, test card numbers). As in the Common Corpus pipeline,
  this keeps text natural, so the model still learns what such values look like
  without learning anyone's real data. The same original always maps to the same
  surrogate, so a document stays internally consistent.
- "tag" (for model output): replace with a visible tag such as [email removed].

Regexes miss some formats and occasionally flag non-personal numbers; this is a
best-effort filter, not a guarantee. Original values are never logged.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable

KINDS = ("email", "phone", "ip", "card", "iban", "ssn")

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w-])")
_PHONE_NANP = re.compile(r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[.-]\d{4}(?![\w])")
_PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?![\w])")
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_CARD = re.compile(r"(?<![\w])(?:\d[ -]?){12,18}\d(?![\w])")
_IBAN = re.compile(r"(?<![\w])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?(?![\w])")
_SSN = re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])")

_RESERVED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
_DOC_IP_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.", "127.0.0.", "0.0.0.")


def _h(value: str, salt: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{salt}:{value}".encode(), digest_size=8).digest(), "big")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1
    except ValueError:
        return False


# --- surrogates (fictitious values from reserved ranges) -------------------------
def _sur_email(v: str) -> str:
    return f"person{_h(v, 'e') % 10000:04d}@example.com"


def _sur_phone(v: str) -> str:
    n = _h(v, "p") % 100
    return f"+1 202-555-01{n:02d}" if v.strip().startswith("+") else f"(202) 555-01{n:02d}"


def _sur_ip(v: str) -> str:
    return f"192.0.2.{_h(v, 'i') % 254 + 1}"


def _sur_card(v: str) -> str:
    return "4242 4242 4242 4242"  # a published test card number, not a real account


def _sur_iban(v: str) -> str:
    return "XX00 0000 0000 0000 0000"


def _sur_ssn(v: str) -> str:
    return "000-00-0000"  # never issued


_SURROGATE: dict[str, Callable[[str], str]] = {
    "email": _sur_email, "phone": _sur_phone, "ip": _sur_ip,
    "card": _sur_card, "iban": _sur_iban, "ssn": _sur_ssn,
}
_TAG = {
    "email": "[email removed]", "phone": "[phone removed]", "ip": "[IP address removed]",
    "card": "[card number removed]", "iban": "[bank account removed]", "ssn": "[ID number removed]",
}


def _detectors() -> list[tuple[str, re.Pattern[str], Callable[[str], bool]]]:
    def digits(s: str) -> str:
        return re.sub(r"\D", "", s)

    return [
        ("email", _EMAIL, lambda v: not v.lower().endswith(_RESERVED_EMAIL_DOMAINS)),
        ("iban", _IBAN, _iban_ok),
        ("card", _CARD, lambda v: 13 <= len(digits(v)) <= 19 and _luhn_ok(digits(v)) and v != "4242 4242 4242 4242"),
        ("ssn", _SSN, lambda v: v != "000-00-0000"),
        ("ip", _IPV4, lambda v: all(int(o) <= 255 for o in v.split(".")) and not v.startswith(_DOC_IP_PREFIXES)),
        ("phone", _PHONE_NANP, lambda v: "555-01" not in v),
        ("phone", _PHONE_INTL, lambda v: 8 <= len(digits(v)) <= 15 and "555-01" not in v),
    ]


_DETECTORS = _detectors()


def scrub(text: str, mode: str = "surrogate", kinds: tuple[str, ...] = KINDS) -> tuple[str, Counter[str]]:
    """Replace personal data in text. Returns the new text and counts per kind (never the values)."""
    if mode not in ("surrogate", "tag"):
        raise ValueError("mode must be 'surrogate' or 'tag'")
    counts: Counter[str] = Counter()
    for kind, pattern, valid in _DETECTORS:
        if kind not in kinds:
            continue

        def replace(m: re.Match[str], kind: str = kind, valid: Callable[[str], bool] = valid) -> str:
            value = m.group(0)
            if not valid(value):
                return value
            counts[kind] += 1
            return _SURROGATE[kind](value) if mode == "surrogate" else _TAG[kind]

        text = pattern.sub(replace, text)
    return text, counts


def contains_pii(text: str) -> bool:
    return bool(scrub(text, mode="tag")[1])
