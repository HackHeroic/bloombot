"""Convert a Claude Design `.dc.html` file into a React view.

    python tools/dc2jsx.py design/BloomBot.dc.html src/views/AppView.jsx
    python tools/dc2jsx.py design/DistBar.dc.html src/components/DistBar.jsx --component DistBar

The template inside <x-dc> becomes a function `view(v)` that returns JSX. `{{ name }}` becomes
`v.name`, `<sc-if>` a conditional, `<sc-for>` a map, `<dc-import>` a component, and every inline
style is kept verbatim through `sx()`. With --component, the design's logic class is appended
as a React class component that renders `view(this.renderVals())`.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

VOID = {"input", "br", "img", "link", "meta", "hr"}
ATTR_RENAME = {"class": "className", "for": "htmlFor", "colspan": "colSpan", "rowspan": "rowSpan",
               "tabindex": "tabIndex", "readonly": "readOnly", "maxlength": "maxLength"}
TOKEN = re.compile(r'<!--.*?-->|<(/?)([a-zA-Z][\w-]*)((?:\s+[\w:@.-]+(?:="[^"]*")?)*)\s*(/?)>|([^<]+)', re.S)
ATTR = re.compile(r'([\w:@.-]+)(?:="([^"]*)")?')
MUSTACHE = re.compile(r"\{\{\s*(.*?)\s*\}\}")
LITERAL = re.compile(r"^(true|false|null|undefined|-?\d+(\.\d+)?)$")


class Node:
    def __init__(self, tag, attrs=None):
        self.tag, self.attrs, self.children = tag, attrs or [], []


def parse(src: str) -> Node:
    root = Node("#root")
    stack = [root]
    for m in TOKEN.finditer(src):
        close, tag, attrs, selfclose, text = m.groups()
        if text is not None:
            stack[-1].children.append(text)
        elif tag is None:  # comment
            continue
        elif close:
            while stack[-1].tag != tag:
                stack.pop()
            stack.pop()
        else:
            node = Node(tag, [(a, v) for a, v in ATTR.findall(attrs)])
            stack[-1].children.append(node)
            if not selfclose and tag not in VOID:
                stack.append(node)
    return root


def expr(e: str, scope: set[str]) -> str:
    e = e.strip()
    head = re.match(r"[A-Za-z_$][\w$]*", e)
    if LITERAL.match(e) or (head and head.group(0) in scope):
        return e
    return "v." + e


def value(raw: str, scope: set[str], style=False) -> str:
    raw = html.unescape(raw)
    parts = MUSTACHE.split(raw)
    if style:
        tpl = "".join(p.replace("`", "\\`").replace("${", "\\${") if i % 2 == 0 else "${" + expr(p, scope) + "}"
                      for i, p in enumerate(parts))
        return "{sx(`" + tpl + "`)}"
    if len(parts) == 3 and parts[0] == "" and parts[2] == "":
        return "{" + expr(parts[1], scope) + "}"
    if len(parts) == 1:
        return "{" + json.dumps(raw, ensure_ascii=False) + "}"
    tpl = "".join(p.replace("`", "\\`").replace("${", "\\${") if i % 2 == 0 else "${" + expr(p, scope) + "}"
                  for i, p in enumerate(parts))
    return "{`" + tpl + "`}"


def camel(name: str) -> str:
    return re.sub(r"-([a-z])", lambda m: m.group(1).upper(), name)


def emit(node, scope: set[str], depth=0) -> str:
    pad = "  " * depth
    if isinstance(node, str):
        if not node.strip() and "\n" in node:
            return ""
        out = []
        for i, p in enumerate(MUSTACHE.split(html.unescape(node))):
            if i % 2:
                out.append("{" + expr(p, scope) + "}")
            elif p:
                out.append("{" + json.dumps(p, ensure_ascii=False) + "}")
        return pad + "".join(out) + "\n"
    attrs = dict(node.attrs)
    kids = lambda sc, d: "".join(emit(c, sc, d) for c in node.children)
    if node.tag == "helmet":
        return ""
    if node.tag == "sc-if":
        cond = expr(MUSTACHE.search(attrs["value"]).group(1), scope)
        return f"{pad}{{{cond} ? (<>\n{kids(scope, depth + 1)}{pad}</>) : null}}\n"
    if node.tag == "sc-for":
        lst = expr(MUSTACHE.search(attrs["list"]).group(1), scope)
        var = attrs["as"]
        inner = scope | {var}
        return (f"{pad}{{({lst} || []).map(({var}, {var}_i) => (<React.Fragment key={{{var}_i}}>\n"
                f"{kids(inner, depth + 1)}{pad}</React.Fragment>))}}\n")
    if node.tag == "dc-import":
        name = attrs.pop("name")
        props = []
        for a, raw in node.attrs:
            if a == "name" or a.startswith("hint-"):
                continue
            props.append(f"{camel(a)}={value(raw or '', scope)}")
        return f"{pad}<{name} {' '.join(props)} />\n"
    tag = node.tag
    props = []
    classes = []
    for a, raw in node.attrs:
        if a.startswith("hint-"):
            continue
        if a == "style-focus":
            classes.append("bb-focus")
            continue
        if a == "style":
            props.append("style=" + value(raw, scope, style=True))
            continue
        name = ATTR_RENAME.get(a, a)
        props.append(f"{name}={value(raw if raw is not None else 'true', scope)}" if raw is not None else name)
    if classes:
        props.append(f'className="{" ".join(classes)}"')
    open_ = f"<{tag}{(' ' + ' '.join(props)) if props else ''}"
    if tag in VOID or not node.children:
        return f"{pad}{open_} />\n"
    return f"{pad}{open_}>\n{kids(scope, depth + 1)}{pad}</{tag}>\n"


def convert(src: str) -> tuple[str, str | None]:
    start, end = src.index("<x-dc>") + len("<x-dc>"), src.rindex("</x-dc>")
    tree = parse(src[start:end])
    body = "".join(emit(c, set(), 2) for c in tree.children)
    m = re.search(r"<script[^>]*data-dc-script[^>]*>(.*?)</script>", src, re.S)
    return body, (m.group(1) if m else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--component", help="also emit the design's logic class as this component")
    ap.add_argument("--imports", default="", help="extra import lines")
    a = ap.parse_args()
    body, logic = convert(Path(a.src).read_text())
    head = ("// GENERATED by tools/dc2jsx.py from " + Path(a.src).name + " — edit the design, not this file.\n"
            "import React from 'react';\nimport { sx } from '../sx.js';\n" + a.imports.replace("\\n", "\n") + "\n")
    view = f"export function view(v) {{\n  return (<>\n{body}  </>);\n}}\n"
    out = head + view
    if a.component:
        cls = logic.replace("class Component extends DCLogic", f"export default class {a.component} extends React.Component")
        cls = cls.rstrip()
        assert cls.endswith("}")
        cls = cls[:-1] + "  render() { return view(this.renderVals()); }\n}\n"
        out += "\n" + cls
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
