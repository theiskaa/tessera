"""Replay allowlisted official HTML captures for approved silver rows.

The profile table is reviewed project code. Lineage can select a profile but
cannot supply extraction code, a capture path, or its own expected hashes.
Capture bodies and response headers are private, ignored files. Preserve them
with the candidate/lineage snapshots; their absence makes approval fail closed.
"""

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re


BLOCK_TAGS = {
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "tr",
    "dt", "dd", "address", "div", "section", "article", "header",
    "blockquote", "pre", "ul", "ol", "dl", "table", "figure", "figcaption",
    "summary", "details", "aside",
}
DROP_TAGS = {"script", "style", "noscript", "template", "svg", "button", "input", "select", "form"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


@dataclass(frozen=True)
class SourceProfile:
    """Pinned identity, capture, and projection for one reviewed source page."""

    profile_id: str
    source: str
    row_id: str
    country: str
    url: str
    raw_sha256: str
    capture_path: str
    capture_sha256: str
    text_sha256: str
    projection: str
    header_path: str = ""
    header_sha256: str = ""


PROFILES = {
    "de_sh_press_q0042_v1": SourceProfile(
        "de_sh_press_q0042_v1", "de-sh",
        "de-sh-www.schleswig-holstein.de_DE_landesregierung_ministerien-behoerden_III_Presse_PI-4397b710",
        "DE", "https://www.schleswig-holstein.de/DE/landesregierung/ministerien-behoerden/III/Presse/PI/2026/08_August/20260826_EVi_Lt?nn=549a8fa0-66c0-4da0-9f19-70e4be245eac",
        "813a497f9b579387795f286b55db4688c6c7f99c1511549554cd3bbe36005696",
        "internal/reports/m7/de-sh-http-capture-v1/q0042.html",
        "21995d7eb730b718d0b2201130e3af1daafebcb5062e17e61db0eacab4c88f20",
        "0434d2744a9783195d69a6b3c751886f0a23d07dd79678e64d747df2021f35ea",
        "de_sh_press_v1", "internal/reports/m7/de-sh-http-capture-v1/q0042.headers.txt",
        "790ef206ee464e965bf1ad234eb219d3cb1fe4d283e18e15cc5d7e716d0c9580"),
    "de_sh_press_p0092_v1": SourceProfile(
        "de_sh_press_p0092_v1", "de-sh",
        "de-sh-www.schleswig-holstein.de_DE_landesregierung_ministerien-behoerden_VI_Presse_PI_-654ff7e9",
        "DE", "https://www.schleswig-holstein.de/DE/landesregierung/ministerien-behoerden/VI/Presse/PI/2026/20260831_Verabschiedung_Uta_Balsies?nn=549a8fa0-66c0-4da0-9f19-70e4be245eac",
        "b89473883465ab3cb9251416e1935a9fd16adfe2a38c69fea9140b77b65a8c3f",
        "internal/reports/m7/de-sh-http-capture-v1/p0092.html",
        "94b1e3d791aab8e5c9290c9b83b298c35c2eb58e57cf98820f95371b63a667cf",
        "dd032f3a9232d886928f63407145a211e9a75e87b56388f62ff40bca7b6457ea",
        "de_sh_press_v1", "internal/reports/m7/de-sh-http-capture-v1/p0092.headers.txt",
        "743d356545a34a7622b19822acf28684dda351707a07b7f5093b4f0343c88706"),
    "ge_eqe_contact_q0237_v1": SourceProfile(
        "ge_eqe_contact_q0237_v1", "ge-structure",
        "ge-structure-www.eqe.ge_ka_contact-f63be75c", "GE",
        "https://www.eqe.ge/ka/contact",
        "2a99a8f7c7a6f4bedf160b0c5848efe7f6234a3500faddcbb22503e3efbf5747",
        "internal/reports/m7/ge-official-contact-captures-20260927-v1/eqe-contact.html",
        "5edf7de8b12e6239336289e89b84d1b36ef4a197cc7816a07c3304c0539d8bf2",
        "05b15b13b5fa52d0175202092c77adae22df2757f5bb526016d771d68ffb4383",
        "ge_eqe_card_v1"),
    "ge_gardabani_article_q0389_v1": SourceProfile(
        "ge_gardabani_article_q0389_v1", "ge-structure",
        "ge-structure-gardabani.gov.ge_gardabani_administrative-units_akhali-samgori-3f964d37",
        "GE", "https://gardabani.gov.ge/gardabani/administrative-units/akhali-samgori/",
        "f6c598e67062fc1519837ce9e9e310fac4577f34a8c13c2473a4ae66a8d866f2",
        "internal/reports/m7/ge-official-contact-captures-20260927-v1/gardabani-akhali-samgori.html",
        "623f285a654088db4c736da63784a91f83d036d68c322b2c5b780f42d7b4ac61",
        "e449d14c969b05940bdd5269421ba76f4ca365b8d4bc31554ca1355de4038f75",
        "ge_gardabani_article_v1"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PressBody(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []
        self.found = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "div" and self.depth == 0:
            classes = (attributes.get("class") or "").split()
            if {"s-richtext", "js-richtext"}.issubset(classes):
                self.depth = 1
                self.found += 1
        elif tag == "div" and self.depth:
            self.depth += 1
        if self.depth:
            if tag == "br":
                self.parts.append("\n")
            elif tag in BLOCK_TAGS:
                self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if self.depth:
            if tag in BLOCK_TAGS:
                self.parts.append("\n\n")
            if tag == "div":
                self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(re.sub(r"\s+", " ", data))


class Node:
    def __init__(self, tag="", attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                self.stack = self.stack[:index]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def normalize_blocks(value):
    paragraphs = []
    for block in re.split(r"\n[ \t]*\n", value):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in block.split("\n")]
        lines = [line for line in lines if line]
        if lines:
            paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


def render(node):
    if isinstance(node, str):
        return re.sub(r"\s+", " ", node)
    if node.tag in DROP_TAGS:
        return ""
    if node.tag == "br":
        return "\n"
    content = "".join(render(child) for child in node.children)
    return f"\n\n{content}\n\n" if node.tag in BLOCK_TAGS else content


def plain_text(node):
    if isinstance(node, str):
        return node
    if node.tag == "br":
        return " "
    content = "".join(plain_text(child) for child in node.children)
    return f" {content} " if node.tag in BLOCK_TAGS else content


def project(raw, projection):
    html = raw.decode("utf-8")
    if projection == "de_sh_press_v1":
        parser = PressBody()
        parser.feed(html)
        if parser.found != 1:
            raise ValueError(f"expected one DE press body, found {parser.found}")
        return normalize_blocks("".join(parser.parts))
    tree = Tree()
    tree.feed(html)
    if projection == "ge_eqe_card_v1":
        matches = [n for n in tree.root.walk() if n.tag == "div" and n.attrs.get("class") == "block block--height box-shadow"]
    elif projection == "ge_gardabani_article_v1":
        matches = [n for n in tree.root.walk() if n.tag == "article" and "post-10022" in n.attrs.get("class", "").split()]
    else:
        raise ValueError("unsupported source projection")
    if len(matches) != 1:
        raise ValueError(f"expected one GE selected scope, found {len(matches)}")
    scope = matches[0]
    output = normalize_blocks(render(scope))
    if (re.sub(r"\s+", " ", plain_text(scope)).strip()
            != re.sub(r"\s+", " ", output).strip()):
        raise ValueError("selected HTML scope omits substantive text")
    if any(re.sub(r"\s+", " ", plain_text(node)).strip()
           for node in scope.walk() if node.tag in DROP_TAGS):
        raise ValueError("selected HTML scope drops nonempty text")
    return output


def verify_source_profile(root: Path, profile_id: str, doc: dict, info: dict) -> SourceProfile:
    """Require exact pinned identity and replayed text for a selected profile."""
    profile = PROFILES.get(profile_id)
    if profile is None:
        raise ValueError("unknown source proof profile")
    if ((doc.get("source"), doc.get("id"), doc.get("country"), info.get("raw_url"),
         info.get("raw_file_sha256"), info.get("candidate_text_sha256"))
            != (profile.source, profile.row_id, profile.country, profile.url,
                profile.raw_sha256, profile.text_sha256)):
        raise ValueError("source proof identity differs from pinned profile")
    capture = (root / profile.capture_path).read_bytes()
    if digest(capture) != profile.capture_sha256:
        raise ValueError("source proof capture SHA mismatch")
    if profile.header_path:
        headers = (root / profile.header_path).read_bytes()
        if (digest(headers) != profile.header_sha256
                or not headers.startswith(b"HTTP/1.1 200 OK\r\n")
                or b"Content-Type: text/html" not in headers):
            raise ValueError("source proof HTTP headers mismatch")
    output = project(capture, profile.projection)
    if digest(output.encode("utf-8")) != profile.text_sha256 or output != doc.get("text"):
        raise ValueError("source proof projection differs from candidate")
    return profile
