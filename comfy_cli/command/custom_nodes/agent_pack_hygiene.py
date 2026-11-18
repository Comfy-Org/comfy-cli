"""Advisory hygiene check for custom node packs the local Comfy agent writes.

The frontend's Settings > Extensions switch skips only an extension's
``setup()`` and node hooks: the module is still imported, so code at module top
level (injected CSS, prototype or draw overrides) keeps running after the user
turns it off. CSS aimed at the whole page restyles every node. And the
"Created by agent" marking needs an ``agent.`` extension name or a
``provenance`` field. This check reports those in a pack's frontend files.

It is a heuristic over the source text, not a JavaScript parser: it finds
statements at brace depth zero, skipping strings, template literals, comments
and regular expressions. Findings are advice and never fail validation.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

INSTALL_RECORD_NAME = ".comfy-agent-install"
MAX_FILE_BYTES = 1 << 20
_WEB_DIRECTORY = re.compile(r"""^\s*WEB_DIRECTORY\s*=\s*["']([^"']+)["']""", re.MULTILINE)
_EXTENSION_NAME = re.compile(r"""registerExtension\s*\(\s*\{[^}]*?\bname\s*:\s*["'`]([^"'`]+)["'`]""", re.DOTALL)
_PROVENANCE = re.compile(r"""\bprovenance\s*:\s*\{[^}]*createdBy\s*:\s*["'`]comfy-agent["'`]""", re.DOTALL)
# Statements a module may run at top level without side effects the switch misses.
_ALLOWED_STATEMENT = re.compile(
    r"""^(?:
        import\b | export\b | (?:async\s+)?function\b | class\b | (?:const|let|var)\b
        | ["']use\ strict["']
        | (?:window\.)?(?:app|comfyApp)\s*\.\s*registerExtension\s*\(
    )""",
    re.VERBOSE,
)
# Selectors that reach the whole page rather than the pack's own elements.
_PAGE_WIDE_SELECTOR = re.compile(
    r"""(?:^|[\s,}>~+(])(\*|html|body|:root|canvas|\.litegraph\b[\w-]*|\.lgraph[\w-]*|\.comfy-[\w-]+)\s*(?=[,{:])"""
)
_LOCAL_DECLARATION = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:class|function|const|let|var)\s+([A-Za-z_$][\w$]*)")
# `X.prop = …` / `X[key] = …` on a class or value the module itself declares.
_LOCAL_ASSIGNMENT = re.compile(r"^([A-Za-z_$][\w$]*)\s*(?:\.[\w$]+|\[[^\]]*\])+\s*=[^=]")
_CSS_BLOCK = re.compile(r"([^{}]+)\{[^{}]*:[^{}]*\}")


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    kind: str
    message: str


def top_level_statements(src: str) -> list[tuple[int, str]]:
    """Statements at brace depth zero, with the 1-based line each starts on."""
    statements: list[tuple[int, str]] = []
    depth = 0
    start = 0
    i = 0
    n = len(src)
    last = ""  # last significant character, to tell a regex from a division

    def flush(end: int) -> None:
        offset = _code_start(src, start, end)
        text = src[offset:end].strip()
        if text:
            statements.append((src.count("\n", 0, offset) + 1, text))

    while i < n:
        c = src[i]
        if c in "\"'":
            i = _skip_string(src, i, c)
            last = c
            continue
        if c == "`":
            i = _skip_template(src, i)
            last = c
            continue
        if src.startswith("//", i):
            end = src.find("\n", i)
            i = n if end == -1 else end
            continue
        if src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c == "/" and (last == "" or last in "(,=:[!&|?{};+-*%<>~^"):
            i = _skip_regex(src, i)
            last = "/"
            continue
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth = max(0, depth - 1)
            if depth == 0 and c == "}" and _ends_block(src, start, i):
                flush(i + 1)
                start = i + 1
                last = ";"
                i += 1
                continue
        elif depth == 0 and (c == ";" or (c == "\n" and _statement_complete(src, start, i))):
            flush(i)
            start = i + 1
            last = ";"
            i += 1
            continue
        if not c.isspace():
            last = c
        i += 1
    flush(n)
    return statements


def _code_start(src: str, i: int, end: int) -> int:
    """Index of the first character in ``src[i:end]`` that is not whitespace or a comment."""
    while i < end:
        if src[i].isspace():
            i += 1
        elif src.startswith("//", i):
            nl = src.find("\n", i)
            i = end if nl == -1 or nl > end else nl + 1
        elif src.startswith("/*", i):
            close = src.find("*/", i + 2)
            i = end if close == -1 or close > end else close + 2
        else:
            break
    return min(i, end)


def _ends_block(src: str, start: int, close: int) -> bool:
    """A `}` at depth zero ends a function or class declaration statement."""
    head = src[_code_start(src, start, close) : close]
    return re.match(r"(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function|class)\b", head) is not None


def _statement_complete(src: str, start: int, newline: int) -> bool:
    """Treat a newline at depth zero as a statement end unless the line obviously continues."""
    text = src[_code_start(src, start, newline) : newline].rstrip()
    if not text:
        return False
    if text[-1] in "=+-*/%&|^!?:,.([{<>":
        return False
    rest = src[newline + 1 :].lstrip()
    return not (rest[:1] in (".", "?", ")", "]", "}", ",") or rest.startswith(("&&", "||", "+", "-", "*", "/")))


def _skip_string(src: str, i: int, quote: str) -> int:
    i += 1
    while i < len(src):
        if src[i] == "\\":
            i += 2
            continue
        if src[i] == quote or src[i] == "\n":
            return i + 1
        i += 1
    return i


def _skip_template(src: str, i: int) -> int:
    i += 1
    while i < len(src):
        if src[i] == "\\":
            i += 2
            continue
        if src[i] == "`":
            return i + 1
        if src.startswith("${", i):
            depth = 1
            i += 2
            while i < len(src) and depth:
                c = src[i]
                if c in "\"'":
                    i = _skip_string(src, i, c)
                    continue
                if c == "`":
                    i = _skip_template(src, i)
                    continue
                depth += c == "{"
                depth -= c == "}"
                i += 1
            continue
        i += 1
    return i


def _skip_regex(src: str, i: int) -> int:
    i += 1
    in_class = False
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "\n":
            return i
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            i += 1
            while i < len(src) and (src[i].isalpha()):
                i += 1
            return i
        i += 1
    return i


def web_directory(pack: Path) -> Path | None:
    """The folder the pack serves frontend files from, per ``WEB_DIRECTORY`` in ``__init__.py``."""
    try:
        init = (pack / "__init__.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _WEB_DIRECTORY.search(init)
    if match is None:
        return None
    web = (pack / match.group(1)).resolve()
    try:
        web.relative_to(pack.resolve())
    except ValueError:
        return None  # Outside the pack: not ours to read.
    return web if web.is_dir() else None


def _frontend_files(pack: Path) -> Iterator[tuple[str, str]]:
    web = web_directory(pack)
    if web is None:
        return
    for path in sorted(web.rglob("*")):
        if path.suffix not in (".js", ".mjs", ".css") or not path.is_file() or path.is_symlink():
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        yield path.relative_to(pack.resolve()).as_posix(), text


def _page_wide_css(rel: str, css: str, base_line: int) -> list[Finding]:
    findings = []
    for block in _CSS_BLOCK.finditer(css):
        for selector in block.group(1).split(","):
            match = _PAGE_WIDE_SELECTOR.search(" " + selector.strip() + " {")
            # A selector that starts from the pack's own element is scoped.
            if match and selector.strip().split()[0] == match.group(1):
                selector_start = block.start(1) + len(block.group(1)) - len(block.group(1).lstrip())
                line = base_line + css.count("\n", 0, selector_start)
                findings.append(
                    Finding(
                        rel,
                        line,
                        "page-wide-css",
                        f"CSS selector `{selector.strip()}` styles the whole page; scope it to the pack's own class",
                    )
                )
                break
    return findings


def _string_literals(src: str) -> Iterator[tuple[int, str]]:
    """(start line, body) of each string and template literal in a JS source."""
    for match in re.finditer(r"`(?:\\.|[^`\\])*`|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", src):
        yield src.count("\n", 0, match.start()) + 1, match.group(0)[1:-1]


def check_pack(pack: Path) -> list[Finding]:
    """Hygiene findings for the pack at ``pack``. Empty when it serves no frontend code."""
    findings: list[Finding] = []
    for rel, text in _frontend_files(pack):
        if rel.endswith(".css"):
            findings += _page_wide_css(rel, text, 1)
            continue
        statements = top_level_statements(text)
        local_names = {m.group(1) for _, stmt in statements if (m := _LOCAL_DECLARATION.match(stmt))}
        for line, statement in statements:
            assignment = _LOCAL_ASSIGNMENT.match(statement)
            if assignment and assignment.group(1) in local_names:
                continue  # Configures the module's own class or value, not the page.
            if not _ALLOWED_STATEMENT.match(statement):
                first = statement.splitlines()[0][:60]
                findings.append(
                    Finding(
                        rel,
                        line,
                        "top-level-code",
                        f"`{first}` runs at module top level, so Settings > Extensions cannot turn it off; "
                        "move it into setup() or a node hook",
                    )
                )
        for line, literal in _string_literals(text):
            if "{" in literal and ":" in literal:
                findings += _page_wide_css(rel, literal, line)
        for name in _EXTENSION_NAME.findall(text):
            if not name.startswith("agent.") and not _PROVENANCE.search(text):
                findings.append(
                    Finding(
                        rel,
                        text.count("\n", 0, text.find(name)) + 1,
                        "extension-name",
                        f"extension `{name}` has no `agent.` prefix and no provenance, "
                        "so the frontend will not mark it as made by the agent",
                    )
                )
    return findings


def is_agent_pack(pack: Path) -> bool:
    """True when the pack carries the agent's install record or an ``agent.`` extension name."""
    if (pack / INSTALL_RECORD_NAME).is_file():
        return True
    return any(
        name.startswith("agent.")
        for rel, text in _frontend_files(pack)
        if not rel.endswith(".css")
        for name in _EXTENSION_NAME.findall(text)
    )
