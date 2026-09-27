"""Replay allowlisted official HTML captures for approved silver rows.

The profile table is reviewed project code. Lineage can select a profile but
cannot supply extraction code, a capture path, or its own expected hashes.
Capture bodies and response headers are private, ignored files. Preserve them
with the candidate/lineage snapshots; their absence makes approval fail closed.
"""

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
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
    source_map_path: str = ""
    source_map_sha256: str = ""
    original_raw_path: str = ""
    original_raw_sha256: str = ""
    original_row_id: str = ""
    original_text_sha256: str = ""
    excerpt_start: int = 0
    excerpt_end: int = 0
    fragment_sha256: str = ""
    publisher_group: str = ""
    required_publisher_groups: tuple[str, ...] = ()
    source_map_parent_group: str = ""


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
        "790ef206ee464e965bf1ad234eb219d3cb1fe4d283e18e15cc5d7e716d0c9580",
        publisher_group="de:sh:ministry_education_science_culture",
        required_publisher_groups=("de:sh:ministry_education_science_culture", "de:schleswig-holstein")),
    "de_sh_press_p0092_v1": SourceProfile(
        "de_sh_press_p0092_v1", "de-sh",
        "de-sh-www.schleswig-holstein.de_DE_landesregierung_ministerien-behoerden_VI_Presse_PI_-654ff7e9",
        "DE", "https://www.schleswig-holstein.de/DE/landesregierung/ministerien-behoerden/VI/Presse/PI/2026/20260831_Verabschiedung_Uta_Balsies?nn=549a8fa0-66c0-4da0-9f19-70e4be245eac",
        "b89473883465ab3cb9251416e1935a9fd16adfe2a38c69fea9140b77b65a8c3f",
        "internal/reports/m7/de-sh-http-capture-v1/p0092.html",
        "94b1e3d791aab8e5c9290c9b83b298c35c2eb58e57cf98820f95371b63a667cf",
        "dd032f3a9232d886928f63407145a211e9a75e87b56388f62ff40bca7b6457ea",
        "de_sh_press_v1", "internal/reports/m7/de-sh-http-capture-v1/p0092.headers.txt",
        "743d356545a34a7622b19822acf28684dda351707a07b7f5093b4f0343c88706",
        publisher_group="de:sh:ministry_finance",
        required_publisher_groups=("de:sh:ministry_finance", "de:schleswig-holstein")),
    "ge_eqe_contact_q0237_v1": SourceProfile(
        "ge_eqe_contact_q0237_v1", "ge-structure",
        "ge-structure-www.eqe.ge_ka_contact-f63be75c", "GE",
        "https://www.eqe.ge/ka/contact",
        "2a99a8f7c7a6f4bedf160b0c5848efe7f6234a3500faddcbb22503e3efbf5747",
        "internal/reports/m7/ge-official-contact-captures-20260927-v1/eqe-contact.html",
        "5edf7de8b12e6239336289e89b84d1b36ef4a197cc7816a07c3304c0539d8bf2",
        "05b15b13b5fa52d0175202092c77adae22df2757f5bb526016d771d68ffb4383",
        "ge_eqe_card_v1", publisher_group="ge:eqe",
        required_publisher_groups=("ge:eqe",)),
    "ge_gardabani_article_q0389_v1": SourceProfile(
        "ge_gardabani_article_q0389_v1", "ge-structure",
        "ge-structure-gardabani.gov.ge_gardabani_administrative-units_akhali-samgori-3f964d37",
        "GE", "https://gardabani.gov.ge/gardabani/administrative-units/akhali-samgori/",
        "f6c598e67062fc1519837ce9e9e310fac4577f34a8c13c2473a4ae66a8d866f2",
        "internal/reports/m7/ge-official-contact-captures-20260927-v1/gardabani-akhali-samgori.html",
        "623f285a654088db4c736da63784a91f83d036d68c322b2c5b780f42d7b4ac61",
        "e449d14c969b05940bdd5269421ba76f4ca365b8d4bc31554ca1355de4038f75",
        "ge_gardabani_article_v1",
        publisher_group="ge:gardabani-municipality",
        required_publisher_groups=("ge:gardabani-municipality",)),
    "ge_q0389_school_excerpt_v1": SourceProfile(
        "ge_q0389_school_excerpt_v1", "ge-structure",
        "ge-q0389-school-address-excerpt-v1", "GE",
        "https://gardabani.gov.ge/gardabani/administrative-units/akhali-samgori/",
        "f4553db09db421c5dcb142327c32b5b0c94ba3500fa7942740ee229f6f5952ca",
        "internal/reports/m7/ge-official-contact-captures-20260927-v1/gardabani-akhali-samgori.html",
        "623f285a654088db4c736da63784a91f83d036d68c322b2c5b780f42d7b4ac61",
        "8904449c729cb6f3661e5c12c81c77ed129a1002540998fbd01350f9ac096f4f",
        "ge_school_excerpt_v1",
        source_map_path="data/interim/silver/ge-q0389-privacy-excerpt-20260927-v1/source-map-v1.jsonl",
        source_map_sha256="87948632ea1386f7e04ef7a9d06948c53385101d63210d4fe81383901efb5d49",
        original_raw_path="data/raw/silver/ge-structure/ge-structure-gardabani.gov.ge_gardabani_administrative-units_akhali-samgori-3f964d37.json",
        original_raw_sha256="f6c598e67062fc1519837ce9e9e310fac4577f34a8c13c2473a4ae66a8d866f2",
        original_row_id="ge-structure-gardabani.gov.ge_gardabani_administrative-units_akhali-samgori-3f964d37",
        original_text_sha256="e449d14c969b05940bdd5269421ba76f4ca365b8d4bc31554ca1355de4038f75",
        excerpt_start=975, excerpt_end=1336,
        publisher_group="ge:gardabani-municipality",
        required_publisher_groups=("ge:gardabani-municipality",)),
    "jp_jma_office_table_v1": SourceProfile(
        "jp_jma_office_table_v1", "jp-jma", "jp-jma-office-table-v1", "JP",
        "https://www.jma.go.jp/jma/kishou/koukai/index1.html",
        "0d01b5731eec91b30ab6c1e00ab56e458b8ac3e32f776aa4ed4d24581f22dd8c",
        "data/raw/jp-new-issuer-pilot-20260927-v1/jma-office.html",
        "afb5b5322632cc90103db5a467aa57ac80ada9137e0f4f9f78e003968e79e5a6",
        "a78b17513cf6816e5170567b4f65c9d4d4edfb28feb1259ca6eab7336bcd4501",
        "jp_jma_table_v1", "data/raw/jp-new-issuer-pilot-20260927-v1/jma-office.headers",
        "8ae333f06099647972f0d351d7c4502c868ce3fa9a6c92c9fa2e00f7192ba4f4",
        "data/interim/silver/jp-new-issuer-pilot-20260927-v1/source-map-v1.jsonl",
        "b3fed32c58ea0bb5c488ca1893205b50d23d5969c9b09ce31e80a24fe5319d5b",
        publisher_group="jp:jma", required_publisher_groups=("jp:jma", "jp:mlit"),
        source_map_parent_group="jp:mlit"),
    "jp_gsi_3d_contact_v1": SourceProfile(
        "jp_gsi_3d_contact_v1", "jp-gsi", "jp-gsi-3d-contact-v1", "JP",
        "https://www.gsi.go.jp/johofukyu/johofukyu60001_00018.html",
        "4ecf7181eaebb45523733b8d675ecd93e57230e0dc877bbace362c65037b4bb3",
        "data/raw/jp-new-issuer-pilot-20260927-v1/gsi-release.html",
        "5429b45e88901326bcabb669d5af44fb4b4ac98517bc980d25f08377bf94445d",
        "2c614ced40bfe3879b960d9310ac732138168d618ffdbf534333981d36e4b082",
        "jp_gsi_contact_v1", "data/raw/jp-new-issuer-pilot-20260927-v1/gsi-release.headers",
        "3a4ca309ceb85e3c9a3bb7bd33a5b9abdd9af477242b5461cc4d27354681960e",
        "data/interim/silver/jp-new-issuer-pilot-20260927-v1/source-map-v1.jsonl",
        "b3fed32c58ea0bb5c488ca1893205b50d23d5969c9b09ce31e80a24fe5319d5b",
        publisher_group="jp:gsi", required_publisher_groups=("jp:gsi", "jp:mlit"),
        source_map_parent_group="jp:mlit"),
    "gb_phs_foi_contact_v1": SourceProfile(
        "gb_phs_foi_contact_v1", "gb-phs-contact", "GBR001", "GB",
        "https://publichealthscotland.scot/contact-us/foi-and-eir-requests/",
        "c7ba848e1400820abcf05290d2651ab09185f85160d1c2c3940e3a70fac77222",
        "data/raw/gb-us-replacement-contact-20260927-v1/gb-phs.html",
        "17c66fa6618f4d05e14f03e7bf3fdba60f4875336f10a98007e5782cf1d6687b",
        "f541f01c031c9e38bda8aea1379fde3f1b8cd5fd1a912df67ec554f5a11df2f3",
        "gb_phs_fragment_v1", "data/raw/gb-us-replacement-contact-20260927-v1/gb-phs.headers",
        "8eea37a8c0b0c11f00936fa47d6775892fc0cd082eceded6105ff08d954e9a84",
        "data/interim/silver/qa-gb-us-replacement-contact-20260927-v1/source-mapping-v1.jsonl",
        "e8ee57419dec375358d37cd565249a0cb40a3835b922bf4cbe5a5fe355f581c1",
        excerpt_start=18146, excerpt_end=19081,
        fragment_sha256="ab1dede9f7343dd9ab37a7514ed15713728a6ecdc15fb51f55945500e45a6c22",
        publisher_group="gb:public_health_scotland",
        required_publisher_groups=("gb:public_health_scotland",)),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def html_200_headers(headers):
    """Check the final captured HTTP response, including HTTP/2 curl headers."""
    responses = [block for block in headers.split(b"\r\n\r\n")
                 if block.startswith(b"HTTP/")]
    if not responses:
        return False
    final = responses[-1]
    status = final.split(b"\r\n", 1)[0]
    return (re.fullmatch(rb"HTTP/(?:1\.1|2) 200(?: .*)?", status) is not None
            and b"content-type: text/html" in final.lower())


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


def one(nodes, description):
    if len(nodes) != 1:
        raise ValueError(f"expected one {description}, found {len(nodes)}")
    return nodes[0]


def checked_projection(scope):
    output = normalize_blocks(render(scope))
    if (re.sub(r"\s+", " ", plain_text(scope)).strip()
            != re.sub(r"\s+", " ", output).strip()):
        raise ValueError("selected HTML scope omits substantive text")
    if any(re.sub(r"\s+", " ", plain_text(node)).strip()
           for node in scope.walk() if node.tag in DROP_TAGS):
        raise ValueError("selected HTML scope drops nonempty text")
    return output


def has_class(node, name):
    return name in node.attrs.get("class", "").split()


def school_excerpt(article):
    item = one([node for node in article.walk() if node.tag == "div"
                and has_class(node, "wd-accordion-item")
                and any(child.tag == "p" and has_class(child, "school-name")
                        for child in node.walk())], "school accordion item")
    title = one([node for node in item.walk() if node.tag == "div"
                 and has_class(node, "wd-accordion-title-text")], "school title")
    content = one([node for node in item.walk() if node.tag == "div"
                   and has_class(node, "wd-accordion-content")], "school content")
    paragraphs = [node for node in content.walk() if node.tag == "p"]
    if len(paragraphs) != 2 or not has_class(paragraphs[0], "school-name"):
        raise ValueError("school paragraph layout changed")
    name, contact = paragraphs
    if (len(contact.children) < 3 or not isinstance(contact.children[0], str)
            or not isinstance(contact.children[1], Node) or contact.children[1].tag != "br"):
        raise ValueError("address is not the first complete contact line")
    heading = checked_projection(title)
    school_name = checked_projection(name)
    address = re.sub(r"\s+", " ", contact.children[0]).strip()
    if heading != "საჯარო სკოლა" or not address.startswith("მისამართი: "):
        raise ValueError("school heading or address changed")
    tail = re.sub(r"\s+", " ", "".join(plain_text(child) for child in contact.children[2:]))
    if not all(word in tail for word in ("დირექტორი:", "ტელ:", "ელ.ფოსტა:")):
        raise ValueError("expected excluded school contact lines missing")
    return "\n\n".join((heading, school_name, address))


class VisibleLines(HTMLParser):
    BLOCKS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ul", "ol",
              "address", "section", "div", "br"}
    SKIP = {"script", "style", "svg", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif not self.skip and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif not self.skip and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(re.sub(r"\s+", " ", data.replace("\xa0", " ")))

    def output(self):
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip()
                 for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def project(raw, projection):
    html = raw.decode("utf-8")
    if projection == "de_sh_press_v1":
        parser = PressBody()
        parser.feed(html)
        if parser.found != 1:
            raise ValueError(f"expected one DE press body, found {parser.found}")
        return normalize_blocks("".join(parser.parts))
    if projection == "gb_phs_fragment_v1":
        parser = VisibleLines()
        parser.feed(html)
        return parser.output()
    tree = Tree()
    tree.feed(html)
    if projection == "ge_eqe_card_v1":
        matches = [n for n in tree.root.walk() if n.tag == "div" and n.attrs.get("class") == "block block--height box-shadow"]
    elif projection == "ge_gardabani_article_v1":
        matches = [n for n in tree.root.walk() if n.tag == "article" and "post-10022" in n.attrs.get("class", "").split()]
    elif projection == "ge_school_excerpt_v1":
        matches = [n for n in tree.root.walk() if n.tag == "article" and has_class(n, "post-10022")]
    elif projection == "jp_jma_table_v1":
        matches = [n for n in tree.root.walk() if n.tag == "table"]
    elif projection == "jp_gsi_contact_v1":
        matches = [n for n in tree.root.walk() if n.tag == "div" and has_class(n, "base_txt")
                   and all(person in plain_text(n) for person in ("阿部", "丹下", "野口", "田村"))]
    else:
        raise ValueError("unsupported source projection")
    scope = one(matches, "selected HTML scope")
    if projection == "ge_school_excerpt_v1":
        return school_excerpt(scope)
    return checked_projection(scope)


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
                or not html_200_headers(headers)):
            raise ValueError("source proof HTTP headers mismatch")
    if profile.source_map_path:
        if ((info.get("derived_source_map_path"), info.get("derived_source_map_sha256"),
             info.get("original_capture_sha256"))
                != (profile.source_map_path, profile.source_map_sha256,
                    profile.capture_sha256)):
            raise ValueError("source proof map identity differs from pinned profile")
        source_map = (root / profile.source_map_path).read_bytes()
        if digest(source_map) != profile.source_map_sha256:
            raise ValueError("source proof map SHA mismatch")
        mappings = [json.loads(line) for line in source_map.splitlines()]
        mapping = one([row for row in mappings if row.get("id") == profile.row_id],
                      "source map row")
        if (mapping.get("url", mapping.get("source_url")) != profile.url
                or mapping.get("capture_path", mapping.get("raw_body_path")) != profile.capture_path
                or mapping.get("capture_sha256", mapping.get("raw_body_sha256")) != profile.capture_sha256
                or mapping.get("headers_path", mapping.get("raw_headers_path", "")) != profile.header_path
                or mapping.get("headers_sha256", mapping.get("raw_headers_sha256", "")) != profile.header_sha256
                or mapping.get("text_sha256", mapping.get("derived_text_sha256")) != profile.text_sha256):
            raise ValueError("source proof map fields differ from pinned profile")
        mapped_publisher = mapping.get("publisher_group", mapping.get("publisher_group_proposed"))
        if profile.publisher_group and mapped_publisher != profile.publisher_group:
            raise ValueError("source proof map publisher differs from pinned profile")
        if (profile.source_map_parent_group
                and mapping.get("publisher_parent") != profile.source_map_parent_group):
            raise ValueError("source proof map parent differs from pinned profile")
    if profile.original_raw_path:
        if (mapping.get("raw_path") != profile.original_raw_path
                or mapping.get("raw_sha256") != profile.original_raw_sha256
                or mapping.get("original_candidate_id") != profile.original_row_id
                or mapping.get("original_text_sha256") != profile.original_text_sha256
                or mapping.get("original_utf8_byte_range")
                != [profile.excerpt_start, profile.excerpt_end]):
            raise ValueError("source proof parent map differs from pinned profile")
        parent_bytes = (root / profile.original_raw_path).read_bytes()
        if digest(parent_bytes) != profile.original_raw_sha256:
            raise ValueError("source proof parent raw SHA mismatch")
        parent = json.loads(parent_bytes)
        parent_text = parent["text"].encode("utf-8")
        if (parent.get("id") != profile.original_row_id or parent.get("url") != profile.url
                or digest(parent_text) != profile.original_text_sha256
                or project(capture, "ge_gardabani_article_v1") != parent["text"]):
            raise ValueError("source proof parent identity or article replay differs")
    if profile.fragment_sha256:
        if (mapping.get("fragment_start_utf8_byte") != profile.excerpt_start
                or mapping.get("fragment_end_utf8_byte") != profile.excerpt_end
                or mapping.get("fragment_sha256") != profile.fragment_sha256):
            raise ValueError("source proof fragment map differs from pinned profile")
        capture = capture[profile.excerpt_start:profile.excerpt_end]
        if digest(capture) != profile.fragment_sha256:
            raise ValueError("source proof fragment SHA mismatch")
    output = project(capture, profile.projection)
    if profile.original_raw_path and output.encode("utf-8") != parent_text[profile.excerpt_start:profile.excerpt_end]:
        raise ValueError("source proof excerpt differs from parent raw bytes")
    if digest(output.encode("utf-8")) != profile.text_sha256 or output != doc.get("text"):
        raise ValueError("source proof projection differs from candidate")
    return profile
