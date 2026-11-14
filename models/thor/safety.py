"""Runtime safety layer for THOR chat.

Small models can't reliably learn safe behavior from training data alone, so
chat adds a second line of defense around the model:

  1. Request screening: messages asking for help with clearly dangerous or
     privacy-violating acts get a fixed, honest response without running the
     model. Messages about self-harm get a supportive message with ways to get help.
  2. Reply screening: the same check runs on the model's reply.
  3. Personal data redaction: emails, phone numbers, card numbers and other
     personal data in replies are replaced with tags (thor/pii.py).

The patterns are deliberately narrow, to avoid refusing ordinary requests
("kill a process", "the movie bombed"). They are a coarse filter, not a
complete safety system; measure them with thor.safety_eval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import pii

_F = re.IGNORECASE


@dataclass(frozen=True)
class Category:
    name: str
    patterns: tuple[re.Pattern[str], ...]
    response: str


CATEGORIES: tuple[Category, ...] = (
    Category(
        "self_harm",
        (
            re.compile(r"\b(kill(ing)?|hurt(ing)?|harm(ing)?|cut(ting)?|starv(e|ing))\s+myself\b", _F),
            re.compile(r"\b(better off without me|no reason to (live|go on))\b", _F),
            re.compile(r"\b(suicide|suicidal|end(ing)? my life|want to die|don'?t want to live)\b", _F),
        ),
        "I'm really sorry you're going through this. I'm a small research model and can't give you the "
        "support a person can, but you don't have to face this alone. If you might act on these thoughts, "
        "please contact your local emergency number now. In the US you can call or text 988 to reach the "
        "988 Suicide & Crisis Lifeline; elsewhere, a local crisis line or someone you trust can help.",
    ),
    Category(
        "weapons",
        (
            re.compile(r"\b(make|build|construct|assemble|synthesi[sz]e|produce|create|manufacture|cook)\b.{0,40}"
                       r"\b(bombs?|explosives?|pipe ?bombs?|ied|detonators?|nerve agents?|sarin|ricin|"
                       r"chemical weapons?|bio-?weapons?|biological weapons?|ghost guns?|untraceable (gun|firearm)s?)\b", _F),
            re.compile(r"\bhow (do|can) i (poison|kill) (a |my )?(person|someone|people|neighbou?r|wife|husband|partner|boss)\b", _F),
            re.compile(r"\b(how (do|can|could|would|to)|ways? to|best way to|help me|tell me how to)\b.{0,20}\bpoison\b", _F),
        ),
        "I can't help with that. Instructions for weapons or for hurting people can cause serious harm. "
        "If someone is in danger, please contact your local emergency services.",
    ),
    Category(
        "cyber_abuse",
        (
            re.compile(r"\b(write|make|create|build|code)\b.{0,30}\b(malware|ransomware|keyloggers?|spyware|"
                       r"computer virus|botnet|credential stealer)\b", _F),
            re.compile(r"\b(hack|break|get) into (someone'?s|somebody'?s|my \w+'?s|(his|her|their))\s?.{0,20}"
                       r"\b(account|email|phone|instagram|facebook|snapchat|tiktok|wi-?fi|laptop|computer)\b", _F),
            re.compile(r"\bsteal (someone'?s |people'?s )?(passwords?|credentials|credit cards?|identity)\b", _F),
        ),
        "I can't help with breaking into accounts or devices, or with writing malicious software. "
        "If you're locked out of your own account, the service's official recovery process is the way back in.",
    ),
    Category(
        "fraud",
        (
            re.compile(r"\b(make|create|forge|print|get|buy|explain how to make)\b.{0,20}\b(fake|forged|counterfeit)\s+"
                       r"(ids?|passports?|driver'?s licen[cs]es?|money|banknotes?|documents?|diplomas?)\b", _F),
        ),
        "I can't help with that. Making or using forged documents or counterfeit money is illegal and hurts people.",
    ),
    Category(
        "privacy",
        (
            re.compile(r"\b(home|current|exact) address of\b", _F),
            re.compile(r"\b(find|look up|track|locate|get)\b.{0,30}\b(where (he|she|they) lives?|someone'?s (home|location|address)"
                       r"|(his|her|their) (home address|phone number|location))\b", _F),
            re.compile(r"\b(social security number|ssn|passport number|bank details|password) (of|for) [A-Z][a-z]+", re.UNICODE),
            re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?'s (social security number|ssn|passport number|bank (account|details)"
                       r"|password|home address|phone number|credit card)", re.UNICODE),
            re.compile(r"\b(phone number|address|email) of (the|that|this|a|my) (girl|boy|woman|man|guy|person|lady|"
                       r"neighbou?r|classmate|teacher|coworker|ex)\b", _F),
            re.compile(r"\bwhere (my|his|her|their) (ex|neighbou?r|classmate|teacher|coworker|boss|crush) lives\b", _F),
            re.compile(r"\bdox(x?ing)?\b", _F),
        ),
        "I can't help find or reveal personal information about someone, such as where they live, their "
        "contact details or their ID numbers. Everyone has a right to privacy.",
    ),
)


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    category: str | None = None
    response: str | None = None


def screen(text: str) -> Verdict:
    for cat in CATEGORIES:
        if any(p.search(text) for p in cat.patterns):
            return Verdict(False, cat.name, cat.response)
    return Verdict(True)


class SafetyGuard:
    def check_request(self, text: str) -> Verdict:
        return screen(text)

    def filter_reply(self, text: str) -> tuple[str, dict]:
        """Screen the model's reply and redact personal data. Returns the safe text and what was changed."""
        verdict = screen(text)
        if not verdict.allowed:
            return verdict.response or "", {"blocked": verdict.category}
        redacted, found = pii.scrub(text, mode="tag")
        return redacted, {"pii_redacted": dict(found)} if found else {}
