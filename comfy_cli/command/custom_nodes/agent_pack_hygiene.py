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

import bisect
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

INSTALL_RECORD_NAME = ".comfy-agent-install"
MAX_FILE_BYTES = 1 << 20
_WEB_DIRECTORY = re.compile(r"""^\s*WEB_DIRECTORY\s*=\s*["']([^"']+)["']\s*(?:#.*)?$""", re.MULTILINE)
_TRIPLE_QUOTED = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_REGISTER = re.compile(r"registerExtension\s*\(\s*\{")
# Initializers that run nothing worth reporting.
_HARMLESS_CALL = re.compile(r"\bnew\s+[A-Za-z_$][\w$.]*\s*\(|\bSymbol\s*\(|\bObject\.freeze\s*\(")
_DECLARATION_RHS = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+[^=]*?=(?!=)(.*)$", re.DOTALL)
_EXPORT_DEFAULT_EXPR = re.compile(r"^export\s+default\s+(?!(?:async\s+)?function\b|class\b)(.*)$", re.DOTALL)
# Statements a module may run at top level without side effects the switch misses.
_ALLOWED_STATEMENT = re.compile(
    r"""^(?:
        import\s*[\w${*"'] | export\b | (?:async\s+)?function\b | class\b | (?:const|let|var)\b
        | ["']use\ strict["']
        | (?:window\.)?(?:app|comfyApp)\s*\.\s*registerExtension\s*\(
    )""",
    re.VERBOSE,
)
# The first compound of a selector that reaches the whole page rather than the pack's own elements.
_PAGE_WIDE_START = re.compile(r"^(?:\*|(?:html|body|canvas)(?![\w-])|:root(?![\w-])|\.litegraph|\.lgraph|\.comfy-)")
_CSS_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
_DECLARED_NAME = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:class|function\*?)\s+([A-Za-z_$][\w$]*)")
_SIMPLE_DECLARATION = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=(?!=)(.*)$", re.DOTALL)
# `X.prop = …` / `X[key] = …` on a class or value the module itself declares.
_LOCAL_ASSIGNMENT = re.compile(r"^([A-Za-z_$][\w$]*)\s*(?:\.[\w$]+|\[[^\]]*\])+\s*=(?!=)(.*)$", re.DOTALL)
# An initializer that creates a new value of the module's own: a literal, object, array, function or class.
_OWN_INITIALIZER = re.compile(
    r"""\s*(?:[{\["'`]|-?\d|true\b|false\b|null\b|new\s|(?:async\s+)?function\b|class\b|(?:async\s*)?\([^()]*\)\s*=>)"""
)
# After these words a `/` starts a regular expression, not a division.
_REGEX_AFTER_WORD = frozenset("return typeof case throw yield in of void delete instanceof new else do await".split())


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    kind: str
    message: str


class _Lines:
    """Line numbers for offsets in one source, without rescanning it each time."""

    def __init__(self, src: str):
        self._starts = [i for i, c in enumerate(src) if c == "\n"]

    def at(self, offset: int) -> int:
        return bisect.bisect_left(self._starts, offset) + 1


def top_level_statements(src: str) -> list[tuple[int, str]]:
    """Statements at brace depth zero, with the 1-based line each starts on."""
    lines = _Lines(src)
    statements: list[tuple[int, str]] = []
    depth = 0
    start = 0
    i = 0
    n = len(src)
    last = ""  # last significant character, to tell a regex from a division
    word_start = -1  # where the identifier ending at `last` began

    def flush(end: int) -> None:
        offset = _code_start(src, start, end)
        text = src[offset:end].strip()
        if text:
            statements.append((lines.at(offset), text))

    while i < n:
        c = src[i]
        if c in "\"'":
            i = _skip_string(src, i, c)
            last, word_start = "x", -1
            continue
        if c == "`":
            i = _skip_template(src, i)
            last, word_start = "x", -1
            continue
        if src.startswith("//", i):
            end = src.find("\n", i)
            i = n if end == -1 else end
            continue
        if src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c == "/":
            after_word = (
                word_start >= 0
                and src[word_start:i].strip() in _REGEX_AFTER_WORD
                and src[:word_start].rstrip()[-1:] != "."
            )
            if last == "" or last in "(,=:[!&|?{};+-*%<>~^" or after_word:
                i = _skip_regex(src, i)
                last, word_start = "x", -1
                continue
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth = max(0, depth - 1)
            if depth == 0 and c == "}" and _ends_block(src, start, i):
                flush(i + 1)
                start = i + 1
                last, word_start = ";", -1
                i += 1
                continue
        elif depth == 0 and (c == ";" or (c == "\n" and _statement_complete(src, start, i, last))):
            flush(i)
            start = i + 1
            last, word_start = ";", -1
            i += 1
            continue
        if c.isalnum() or c in "_$":
            if not (last.isalnum() or last in "_$") or word_start < 0:
                word_start = i
        elif not c.isspace():
            word_start = -1
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


def _statement_complete(src: str, start: int, newline: int, last: str) -> bool:
    """Treat a newline at depth zero as a statement end unless the code obviously continues.

    ``last`` is the last code character before the newline (comments excluded).
    """
    if _code_start(src, start, newline) >= newline:
        return False
    if last in "=+-*/%&|^!?:,.([{<>":
        return False
    nxt = _code_start(src, newline + 1, len(src))
    rest = src[nxt : nxt + 2]
    return not (rest[:1] in (".", "?", ")", "]", "}", ",", ":", "{") or rest in ("&&", "||") or rest[:1] in "+-*")


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
    """Index just past the template literal starting at ``i``; nested templates without recursion."""
    stack = [-1]  # -1: inside template text; n >= 0: inside `${`, at brace depth n
    i += 1
    while i < len(src) and stack:
        c = src[i]
        if stack[-1] < 0:
            if c == "\\":
                i += 2
                continue
            if c == "`":
                stack.pop()
                i += 1
                continue
            if src.startswith("${", i):
                stack.append(0)
                i += 2
                continue
            i += 1
            continue
        if c in "\"'":
            i = _skip_string(src, i, c)
        elif c == "`":
            stack.append(-1)
            i += 1
        elif c == "{":
            stack[-1] += 1
            i += 1
        elif c == "}":
            if stack[-1] == 0:
                stack.pop()
            else:
                stack[-1] -= 1
            i += 1
        else:
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


def web_directory(pack: Path, notes: list[str] | None = None) -> Path | None:
    """The folder the pack serves frontend files from, per ``WEB_DIRECTORY`` in ``__init__.py``.

    ``notes`` collects why nothing could be scanned, so the caller never reports an all-clear for it.
    """
    notes = notes if notes is not None else []
    init_path = pack / "__init__.py"
    if not init_path.exists():
        return None
    try:
        init = init_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        notes.append(f"__init__.py could not be read ({e.strerror or e}), so the frontend code was not scanned")
        return None
    match = _WEB_DIRECTORY.search(_TRIPLE_QUOTED.sub("", init))
    if match is None:
        if "WEB_DIRECTORY" in init:
            notes.append("WEB_DIRECTORY in __init__.py is not a plain string, so the frontend code was not scanned")
        return None
    try:
        web = (pack / match.group(1)).resolve()
        web.relative_to(pack.resolve())
        if web.is_dir():
            return web
        notes.append(f"WEB_DIRECTORY {match.group(1)!r} is not a folder, so nothing was scanned")
        return None
    except ValueError:
        notes.append(f"WEB_DIRECTORY {match.group(1)!r} points outside the pack, so it was not scanned")
        return None
    except (OSError, RuntimeError):
        notes.append(f"WEB_DIRECTORY {match.group(1)!r} could not be resolved (a symlink loop?), so it was not scanned")
        return None


def _frontend_files(pack: Path, notes: list[str] | None = None) -> Iterator[tuple[str, str]]:
    notes = notes if notes is not None else []
    web = web_directory(pack, notes)
    if web is None:
        return
    failures: list[OSError] = []
    paths = sorted(
        Path(folder) / name for folder, _dirs, files in os.walk(web, onerror=failures.append) for name in files
    )
    for failure in failures:
        notes.append(f"could not list {failure.filename or 'a folder'}: {failure.strerror or failure}")
    root = pack.resolve()
    for path in paths:
        if path.suffix not in (".js", ".mjs", ".css"):
            continue
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
        if path.is_symlink() or not path.is_file():
            notes.append(f"skipped {rel}: a symlink or not a regular file")
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                notes.append(f"skipped {rel}: larger than 1 MiB")
                continue
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            notes.append(f"skipped {rel}: unreadable or not UTF-8")
            continue
        yield rel, text


def _css_rules(css: str) -> Iterator[tuple[int, str]]:
    """(offset, selector text) of each rule whose body holds a declaration; one linear pass."""
    start = 0
    i = 0
    while i < len(css):
        c = css[i]
        if c == "{":
            close = css.find("}", i)
            nested = css.find("{", i + 1)
            if close == -1:
                return
            if nested != -1 and nested < close:  # a wrapper (@media, or a rule with nested rules)
                own = css[i + 1 : nested]
                if not css[start:i].lstrip().startswith("@") and re.search(r":[^;{}]*;", own):
                    yield start, css[start:i]  # the parent's own declarations apply to it
                start = i + 1
                i += 1
                continue
            if ":" in css[i:close]:
                yield start, css[start:i]
            start = i = close + 1
            continue
        if c in "};":
            start = i + 1
        i += 1


def _page_wide_css(rel: str, css: str, base_line: int) -> list[Finding]:
    findings = []
    css = _CSS_COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), css)
    lines = _Lines(css)
    for offset, selectors in _css_rules(css):
        for selector in selectors.split(","):
            selector = selector.strip()
            # A selector that starts from the pack's own element is scoped; one that starts at
            # the page (or the shared LiteGraph / Comfy UI classes) reaches every node.
            if selector and _PAGE_WIDE_START.match(selector):
                lead = len(selectors) - len(selectors.lstrip())
                findings.append(
                    Finding(
                        rel,
                        base_line + lines.at(offset + lead) - 1,
                        "page-wide-css",
                        f"CSS selector `{selector}` styles the whole page; scope it to the pack's own class",
                    )
                )
                break
    return findings


def _string_literals(src: str) -> Iterator[tuple[int, str]]:
    """(start line, body) of each string and template literal in a JS source."""
    lines = _Lines(src)
    for match in re.finditer(r"`(?:\\.|[^`\\])*`|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", src):
        yield lines.at(match.start()), match.group(0)[1:-1]


def _blank_nested(src: str) -> str:
    """``src`` with every string, template and bracketed group blanked out, keeping offsets."""
    out = list(src)
    i = 0
    while i < len(src):
        c = src[i]
        if c in "\"'`":
            end = _skip_string(src, i, c) if c != "`" else _skip_template(src, i)
            out[i:end] = " " * (end - i)
            i = end
            continue
        if c in "{[":
            close, depth, j = {"{": "}", "[": "]"}[c], 0, i
            while j < len(src):
                if src[j] in "\"'`":
                    j = _skip_string(src, j, src[j]) if src[j] != "`" else _skip_template(src, j)
                    continue
                depth += src[j] == c
                depth -= src[j] == close
                j += 1
                if depth == 0:
                    break
            out[i:j] = " " * (j - i)
            i = j
            continue
        i += 1
    return "".join(out)


def _runs_code(rhs: str) -> bool:
    """True when a declaration's initializer calls something at import time."""
    flat = _HARMLESS_CALL.sub(" ", _blank_nested(rhs))
    if re.match(r"\s*(?:async\s*)?(?:\([^()]*\)|[\w$]+)\s*=>", flat):
        return False  # the whole initializer is an arrow function; its body runs later
    flat = re.sub(r"(?:\([^()]*\)|[\w$]+)\s*=>", " ", flat)  # inner arrows' parameters
    flat = re.sub(r"\bfunction\b[^(]*\([^)]*\)", " ", flat)  # a function expression's parameters
    return re.search(r"[\w$)\]]\s*\(", flat) is not None


def _matching_close(text: str, start: int) -> int:
    """Index just past the bracket that closes the one at ``start``, skipping strings and comments."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c in "\"'`":
            i = _skip_string(text, i, c) if c != "`" else _skip_template(text, i)
            continue
        if text.startswith(("//", "/*"), i):
            i = _code_start(text, i, len(text))
            continue
        depth += c in "{(["
        depth -= c in "})]"
        i += 1
        if depth == 0:
            return i
    return len(text)


def _top_level_keys(body: str) -> dict[str, tuple[int, str]]:
    """``key -> (offset, raw value text)`` for the top-level keys of an object literal ``body``."""
    keys: dict[str, tuple[int, str]] = {}
    i = 1
    expect_key = True
    while i < len(body) - 1:
        i = _code_start(body, i, len(body) - 1)
        if i >= len(body) - 1:
            break
        c = body[i]
        if expect_key:
            key = re.match(r"""([A-Za-z_$][\w$]*|"[^"]*"|'[^']*')\s*:""", body[i:])
            if key:
                name = key.group(1).strip("\"'")
                value_start = _code_start(body, i + key.end(), len(body) - 1)
                value_end = _value_end(body, value_start)
                keys.setdefault(name, (value_start, body[value_start:value_end]))
                i = value_end
                expect_key = False
                continue
        if c == ",":
            expect_key = True
            i += 1
            continue
        i = _value_end(body, i)
        expect_key = False
    return keys


def _value_end(body: str, i: int) -> int:
    """Index of the ``,`` (or closing brace) that ends the value starting at ``i``."""
    end = len(body) - 1
    while i < end:
        c = body[i]
        if c in "\"'`":
            i = _skip_string(body, i, c) if c != "`" else _skip_template(body, i)
        elif body.startswith(("//", "/*"), i):
            i = _code_start(body, i, end)
        elif c in "{[(":
            i = _matching_close(body, i)
        elif c == ",":
            return i
        else:
            i += 1
    return end


def _registrations(text: str) -> Iterator[tuple[int, str, bool]]:
    """(offset, name, has agent provenance) for each ``registerExtension({...})`` with a top-level name."""
    for match in _REGISTER.finditer(text):
        start = match.end() - 1
        body = text[start : _matching_close(text, start)]
        keys = _top_level_keys(body)
        if "name" not in keys:
            continue
        offset, raw = keys["name"]
        literal = re.fullmatch(r"""\s*["'`]([^"'`]*)["'`]\s*""", _CSS_COMMENT.sub(" ", re.sub(r"//[^\n]*", " ", raw)))
        if literal is None:
            continue
        provenance = keys.get("provenance", (0, ""))[1].strip()
        created_by = _top_level_keys(provenance).get("createdBy", (0, ""))[1] if provenance.startswith("{") else ""
        has_provenance = re.fullmatch(r"""\s*["'`]comfy-agent["'`]\s*""", created_by) is not None
        yield start + offset, literal.group(1), has_provenance


def _own_names(statements: list[tuple[int, str]]) -> set[str]:
    """Classes, functions and plain values the module declares itself (not aliases of globals)."""
    names = set()
    for _, stmt in statements:
        if m := _DECLARED_NAME.match(stmt):
            names.add(m.group(1))
        elif (m := _SIMPLE_DECLARATION.match(stmt)) and _OWN_INITIALIZER.match(m.group(2)):
            if not _runs_code(m.group(2)) and "." not in _blank_nested(m.group(2)):
                names.add(m.group(1))
    return names


def _top_level_finding(statement: str, own: set[str]) -> bool:
    """True when ``statement`` runs code at import time that the extension switch cannot stop."""
    if (assignment := _LOCAL_ASSIGNMENT.match(statement)) and assignment.group(1) in own:
        return _runs_code(assignment.group(2))
    if (declaration := _DECLARATION_RHS.match(statement)) and _runs_code(declaration.group(1)):
        return True
    if (default := _EXPORT_DEFAULT_EXPR.match(statement)) and _runs_code(default.group(1)):
        return True
    return not _ALLOWED_STATEMENT.match(statement)


def check_pack(pack: Path, notes: list[str] | None = None) -> list[Finding]:
    """Hygiene findings for the pack at ``pack``. Empty when it serves no frontend code.

    Anything that could not be scanned is appended to ``notes``.
    """
    findings: list[Finding] = []
    for rel, text in _frontend_files(pack, notes):
        if rel.endswith(".css"):
            findings += _page_wide_css(rel, text, 1)
            continue
        statements = top_level_statements(text)
        own = _own_names(statements)
        for line, statement in statements:
            if _top_level_finding(statement, own):
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
        lines = _Lines(text)
        for offset, name, has_provenance in _registrations(text):
            if not name.startswith("agent.") and not has_provenance:
                findings.append(
                    Finding(
                        rel,
                        lines.at(offset),
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
        for _, name, _ in _registrations(text)
    )
