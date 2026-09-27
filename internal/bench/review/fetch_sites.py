"""Real documents for Georgia and Japan, for the 7.5 evaluation sets and silver training text.

    python internal/bench/review/fetch_sites.py SITE [limit]

Each site lists article pages (from index pages or a range of sequential ids), extracts the
main text, and keeps pages with enough text (and a phone number where `contact` is set).
Evaluation sites write to data/raw/review/<site>/, training sites to data/raw/silver/<site>/.
Pages are fetched three at a time with a 0.75 s pause, with the bench User-Agent, and a
resumed run skips pages already on disk. Oversized extracted text is rejected rather than
joining nonadjacent sections; existing saved pages still require separate source review.
"""

import datetime
import hashlib
import importlib
import re
import sys
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

import common

TODAY = datetime.date.today().isoformat()
JP_GOV = ("Japanese government standard terms of use 2.0 (compatible with CC BY 4.0)",
          "https://www.digital.go.jp/copyright-policy")
JP_CITY = ("City of Yokohama terms of use (government standard terms, compatible with CC BY 4.0)",
           "https://www.city.yokohama.lg.jp/aboutweb/site/policy/copyright.html")
GE_EVAL = ("public web page, evaluation use only, never redistributed", "per site")
GE_GOV = ("official publication of a Georgian public body (Law of Georgia on Copyright and "
          "Neighbouring Rights, art. 8: official documents are not protected)",
          "https://matsne.gov.ge/ka/document/view/13017")

DE_EVAL = ("public web page, evaluation use only, never redistributed", "per site")
GB_EVAL = DE_EVAL
GB_GOV = ("Open Government Licence v3.0",
          "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/")
DE_GOV = ("amtliches Werk of a German federal authority (§ 5 UrhG: official works are not "
          "protected by copyright)", "https://www.gesetze-im-internet.de/urhg/__5.html")
DE_POSTCODE = re.compile(r"\b\d{5} [A-ZÄÖÜ][a-zäöüß]")

SITES = {
    "jp-soumu": dict(use="eval", country="JP", doc_type="press release", licence=JP_GOV,
                     seeds=[f"https://www.soumu.go.jp/menu_news/s-news/{y}{m:02d}m.html"
                            for y in (26, 25) for m in range(12, 0, -1)],
                     link=r"/menu_news/s-news/(?!\d{4}m\.html)[\w-]+\.html$", contact=True),
    "jp-maff": dict(use="eval", country="JP", doc_type="press release", licence=JP_GOV,
                    seeds=["https://www.maff.go.jp/j/press/index.html",
                           "https://www.maff.go.jp/j/press/arc/index.html"],
                    link=r"maff\.go\.jp/.*press/.*\d{6}(_\d+)?\.html$", contact=True),
    "jp-caa": dict(use="eval", country="JP", doc_type="press release", licence=JP_GOV,
                   seeds=["https://www.caa.go.jp/notice/", "https://www.caa.go.jp/notice/archive/"],
                   link=r"/notice/entry/\d+/?$", contact=True),
    "jp-digital": dict(use="eval", country="JP", doc_type="press release", licence=JP_GOV,
                       seeds=["https://www.digital.go.jp/press"]
                       + [f"https://www.digital.go.jp/press?page={n}" for n in range(1, 6)],
                       link=r"digital\.go\.jp/news/[0-9a-f-]{36}$", contact=True),
    "jp-env": dict(use="silver", country="JP", doc_type="press release", licence=JP_GOV,
                   seeds=["https://www.env.go.jp/press/index.html"],
                   link=r"/press/(press_\d+|\d+_\d+|\d+)\.html$", contact=True),
    "ge-economy": dict(use="eval", country="GE", doc_type="news", licence=GE_EVAL,
                       pages=[f"https://www.economy.ge/?page=news&nw={n}" for n in range(3240, 2900, -1)],
                       contact=False),
    "ge-nbg": dict(use="eval", country="GE", doc_type="news", licence=GE_EVAL,
                   seeds=["https://nbg.gov.ge/media/news"]
                   + [f"https://nbg.gov.ge/media/news?page={n}" for n in range(2, 6)],
                   link=r"nbg\.gov\.ge/media/news/[\w-]+$", contact=False),
    "ge-civil": dict(use="eval", country="GE", doc_type="news", licence=GE_EVAL,
                     seeds=["https://civil.ge/archives/category/news"]
                     + [f"https://civil.ge/archives/category/news/page/{n}" for n in range(2, 8)],
                     link=r"civil\.ge/archives/\d+$", contact=False),
    "ge-tsu": dict(use="eval", country="GE", doc_type="news", licence=GE_EVAL,
                   seeds=["https://www.tsu.ge/ka/news"]
                   + [f"https://www.tsu.ge/ka/news?page={n}" for n in range(2, 6)],
                   link=r"tsu\.ge/ka/news/[^?]{8,}$", contact=False),
    "ge-contacts": dict(use="eval", country="GE", doc_type="contact page", licence=GE_EVAL,
                        pages=[
                            "https://mof.ge/ka/Contact", "https://www.nbg.gov.ge/page/contact",
                            "https://www.economy.ge/?page=contacts", "https://www.geostat.ge/ka/contact",
                            "https://www.tsu.ge/ka/contact", "https://iliauni.edu.ge/ge/contact",
                            "https://gtu.ge/contact", "https://freeuni.edu.ge/ka/contact",
                            "https://www.btu.edu.ge/ka/contact", "https://www.cu.edu.ge/ka/contact",
                            "https://www.ibsu.edu.ge/ka/contact", "https://www.sangu.edu.ge/ka/contact",
                            "https://www.tbcbank.ge/web/ka/contact", "https://bankofgeorgia.ge/ka/contact",
                            "https://www.libertybank.ge/ka/contact", "https://www.magticom.ge/ka/contact",
                            "https://www.silknet.com/ka/contact", "https://www.gwp.ge/ka/contact",
                            "https://www.telasi.ge/ka/contact", "https://www.gpc.ge/ka/contact",
                            "https://www.psp.ge/ka/contact", "https://www.nikora.ge/ka/contact",
                            "https://www.aversi.ge/ka/contact", "https://batumi.ge/ka/contact",
                            "https://kutaisi.gov.ge/ka/contact",
                            "https://sao.ge/contact", "https://www.gncc.ge/ka/contact",
                            "https://sda.gov.ge/?page_id=2",
                            "https://www.enterprisegeorgia.gov.ge/ka/contact",
                            "https://www.georgia.travel/contact", "https://www.gita.gov.ge/geo/contact",
                            "https://www.moe.gov.ge/ka/contact", "https://police.ge/ge/contact",
                            "https://www.ssa.gov.ge/contact",
                        ], contact=True),
    "ge-tbilisi": dict(use="silver", country="GE", doc_type="news", licence=GE_GOV,
                       pages=[f"http://tbilisi.gov.ge/news/{n}" for n in range(9802, 9300, -1)],
                       contact=False),
    "ge-parliament": dict(use="silver", country="GE", doc_type="news", licence=GE_GOV,
                          seeds=["https://www.parliament.ge/media/news"]
                          + [f"https://www.parliament.ge/media/news?page={n}" for n in range(2, 10)],
                          link=r"parliament\.ge/media/news/(?!category)[\w-]{6,}$", contact=False),
    "de-berlin": dict(use="eval", country="DE", doc_type="press release", licence=DE_EVAL,
                      seeds=["https://www.berlin.de/presse/pressemitteilungen/"]
                      + [f"https://www.berlin.de/presse/pressemitteilungen/index/index/page/{n}?searchtext="
                         for n in range(2, 12)],
                      link=r"berlin\.de/.*/pressemitteilung\.\d+\.php$", contact="any"),
    "de-impressum": dict(use="eval", country="DE", doc_type="imprint", licence=DE_EVAL,
                         pages=['https://www.dm.de/impressum', 'https://www.rewe.de/impressum/', 'https://www.bahn.de/impressum', 'https://www.telekom.de/impressum', 'https://www.vodafone.de/impressum.html', 'https://www.adac.de/impressum/', 'https://www.tagesschau.de/impressum', 'https://www.spiegel.de/impressum', 'https://www.zeit.de/impressum/index', 'https://www.sueddeutsche.de/impressum', 'https://www.heise.de/impressum.html', 'https://www.golem.de/sonstiges/impressum.html', 'https://www.otto.de/shoppages/service/impressum', 'https://www.check24.de/unternehmen/impressum/', 'https://www.idealo.de/impressum.html', 'https://www.immobilienscout24.de/impressum.html', 'https://www.commerzbank.de/impressum/', 'https://www.dkb.de/impressum/', 'https://www.ing.de/impressum/', 'https://www.comdirect.de/impressum.html', 'https://www.allianz.de/impressum/', 'https://www.huk.de/impressum.html', 'https://www.barmer.de/impressum', 'https://www.bvg.de/de/impressum', 'https://www.hvv.de/de/impressum', 'https://www.dwd.de/DE/service/impressum/impressum_node.html', 'https://www.dekra.de/de/impressum/', 'https://www.fraunhofer.de/de/impressum.html', 'https://www.mpg.de/impressum', 'https://www.helmholtz.de/impressum/', 'https://www.tum.de/impressum', 'https://www.uni-heidelberg.de/de/impressum', 'https://www.hu-berlin.de/de/hu/impressum', 'https://www.fu-berlin.de/redaktion/impressum/index.html', 'https://www.kit.edu/impressum.php', 'https://www.dfb.de/impressum/', 'https://www.thalia.de/impressum', 'https://www.chefkoch.de/impressum.html', 'https://www.gelbeseiten.de/impressum', 'https://www.verbraucherzentrale.de/impressum', 'https://www.test.de/Impressum-1244634-0/', 'https://www.miele.de/de/m/impressum-1386.htm', 'https://www.tuev-nord.de/de/impressum/', 'https://www.ndr.de/service/impressum/index.html', 'https://www.wdr.de/impressum/index.html', 'https://www.br.de/unternehmen/service/impressum/index.html', 'https://www.zdf.de/zdfunternehmen/impressum-100.html', 'https://www.t-online.de/impressum/', 'https://www.welt.de/services/article7893735/Impressum.html', 'https://www.handelsblatt.com/impressum/', 'https://www.stern.de/impressum-3108480.html', 'https://www.focus.de/intern/impressum/', 'https://www.kicker.de/impressum', 'https://www.aldi-nord.de/impressum.html', 'https://www.aldi-sued.de/de/impressum.html', 'https://www.edeka.de/impressum.jsp', 'https://www.rossmann.de/de/impressum', 'https://www.tchibo.de/impressum-c400078524.html', 'https://www.hornbach.de/impressum/', 'https://www.obi.de/impressum', 'https://www.bauhaus.info/impressum', 'https://www.deutsche-rentenversicherung.de/DRV/DE/Impressum/impressum_node.html', 'https://www.arbeitsagentur.de/impressum', 'https://www.bundesregierung.de/breg-de/service/impressum', 'https://www.bundestag.de/impressum', 'https://www.berlin.de/impressum/', 'https://www.hamburg.de/impressum/', 'https://www.muenchen.de/impressum', 'https://www.stadt-koeln.de/service/impressum/index.html', 'https://www.frankfurt.de/impressum', 'https://www.stuttgart.de/impressum.php', 'https://www.duesseldorf.de/impressum', 'https://www.leipzig.de/impressum', 'https://www.dresden.de/de/impressum.php', 'https://www.hannover.de/Impressum', 'https://www.nuernberg.de/internet/portal/impressum.html', 'https://www.bremen.de/impressum'], contact="any"),
    "gb-contacts": dict(use="eval", country="GB", doc_type="contact page", licence=GB_EVAL,
                        pages=['https://www.cancerresearchuk.org/about-us/contact-us', 'https://www.oxfam.org.uk/contact-us/', 'https://www.nspcc.org.uk/about-us/contact-us/', 'https://www.redcross.org.uk/contact-us', 'https://www.macmillan.org.uk/about-us/contact-us', 'https://www.barnardos.org.uk/contact-us', 'https://www.bhf.org.uk/contact-us', 'https://www.ageuk.org.uk/contact-us/', 'https://www.mind.org.uk/about-us/contact-us/', 'https://www.shelter.org.uk/contact_us', 'https://www.nationaltrust.org.uk/contact-us', 'https://www.rnli.org/contact-us', 'https://www.samaritans.org/how-we-can-help/contact-samaritan/', 'https://www.citizensadvice.org.uk/about-us/contact-us/', 'https://www.which.co.uk/help/contact-us', 'https://www.ox.ac.uk/contact-us', 'https://www.ucl.ac.uk/contact-ucl', 'https://www.imperial.ac.uk/contact/', 'https://www.kcl.ac.uk/contact', 'https://www.manchester.ac.uk/contact/', 'https://www.ed.ac.uk/contact', 'https://www.gla.ac.uk/contact/', 'https://www.bristol.ac.uk/contact/', 'https://www.leeds.ac.uk/contact', 'https://www.sheffield.ac.uk/contact', 'https://www.birmingham.ac.uk/contact', 'https://www.nottingham.ac.uk/contact', 'https://www.southampton.ac.uk/contact', 'https://www.lse.ac.uk/contact-us', 'https://www.qmul.ac.uk/contact/', 'https://www.bbc.co.uk/contact', 'https://www.theguardian.com/help/contact-us', 'https://www.ons.gov.uk/aboutus/contactus', 'https://www.nhs.uk/contact-us/', 'https://www.metoffice.gov.uk/about-us/contact', 'https://www.tfl.gov.uk/help-and-contact/', 'https://www.royalmail.com/contact-us', 'https://www.britishmuseum.org/about-us/contact-us', 'https://www.nhm.ac.uk/about-us/contact-us.html', 'https://www.tate.org.uk/about-us/contact-us', 'https://www.leeds.gov.uk/contact-us', 'https://www.bristol.gov.uk/council-and-mayor/contact-us', 'https://www.manchester.gov.uk/contact', 'https://www.birmingham.gov.uk/contact', 'https://www.edinburgh.gov.uk/contact-us', 'https://www.glasgow.gov.uk/contactus', 'https://www.cardiff.gov.uk/ENG/Contact-us/', 'https://www.sheffield.gov.uk/contact-us', 'https://www.nottinghamcity.gov.uk/contact-us', 'https://www.liverpool.gov.uk/contact-us/', 'https://www.oxford.gov.uk/contact', 'https://www.cambridge.gov.uk/contact-us', 'https://www.york.gov.uk/contact-us', 'https://www.brighton-hove.gov.uk/contact-us', 'https://www.kent.gov.uk/about-the-council/contact-us', 'https://www.surreycc.gov.uk/council-and-democracy/contact-us', 'https://www.hertfordshire.gov.uk/contact-us/contact-us.aspx', 'https://www.cornwall.gov.uk/contact-us/', 'https://www.devon.gov.uk/help/contact-us/', 'https://www.norfolk.gov.uk/contact-us', 'https://www.lancashire.gov.uk/council/contact-us/', 'https://www.rspb.org.uk/contact-us', 'https://www.wwf.org.uk/contact-us', 'https://www.stjohnambulance.org.uk/contact-us', 'https://www.actionforchildren.org.uk/contact-us/', 'https://www.scope.org.uk/contact-us/', 'https://www.sightsavers.org/contact-us/', 'https://www.guidedogs.org.uk/contact-us/', 'https://www.mariecurie.org.uk/about/contact-us', 'https://www.alzheimers.org.uk/about-us/contact-us'], contact="any"),
    "gb-courts": dict(use="eval", country="GB", doc_type="contact page", licence=GB_GOV,
                      pages=[f"https://www.find-court-tribunal.service.gov.uk/courts/{c}-{kind}"
                             for c in ['birmingham', 'manchester', 'leeds', 'liverpool', 'bristol', 'sheffield', 'nottingham', 'leicester', 'newcastle', 'cardiff', 'swansea', 'plymouth', 'exeter', 'norwich', 'cambridge', 'oxford', 'reading', 'southampton', 'portsmouth', 'brighton', 'canterbury', 'chelmsford', 'luton', 'bradford', 'hull', 'york', 'derby', 'stoke-on-trent', 'wolverhampton', 'coventry', 'northampton', 'peterborough', 'ipswich', 'lincoln', 'carlisle', 'preston', 'blackburn', 'bolton', 'wigan', 'truro']
                             for kind in ("crown-court", "magistrates-court",
                                          "county-court-and-family-court")],
                      contact="any"),
    "de-destatis": dict(use="silver", country="DE", doc_type="press release", licence=DE_GOV,
                        seeds=["https://www.destatis.de/DE/Presse/Pressemitteilungen/_inhalt.html"],
                        link=r"destatis\.de/DE/Presse/Pressemitteilungen/\d{4}/\d{2}/PD\w+\.html$",
                        contact="any"),
    "de-bnetza": dict(use="silver", country="DE", doc_type="press release", licence=DE_GOV,
                      seeds=["https://www.bundesnetzagentur.de/DE/Allgemeines/Presse/Pressemitteilungen/start.html"]
                      + [f"https://www.bundesnetzagentur.de/DE/Allgemeines/Presse/Pressemitteilungen/start.html?gtp=859560_list%253D{n}"
                         for n in range(2, 14)],
                      link=r"bundesnetzagentur\.de/SharedDocs/Pressemitteilungen/DE/20\d\d/\w+\.html$",
                      contact=False),
    "ge-tbilisi2": dict(use="silver", country="GE", doc_type="news", licence=GE_GOV,
                        pages=[f"http://tbilisi.gov.ge/news/{n}" for n in range(9300, 8700, -1)],
                        contact=False),
    "ge-mepa": dict(use="silver", country="GE", doc_type="news", licence=GE_GOV,
                    pages=[f"https://mepa.gov.ge/Ge/News/Details/{n}" for n in range(27607, 27000, -1)],
                    contact=False),
}
# Training-only sites added per country, each in its own module (sites_de.py, ...).
for extra in ("sites_de", "sites_gb", "sites_ge", "sites_jp", "sites_us", "sites_eval"):
    try:
        SITES.update(importlib.import_module(extra).SITES)
    except ModuleNotFoundError:
        pass
MIN_CHARS = 400


def host(url):
    """Canonical publisher host; `www` is not a separate publisher."""
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def source_hosts():
    """Declared publisher hosts for each use and country."""
    uses = {}
    all_uses = {}
    for name, site in SITES.items():
        for url in list(site.get("pages", ())) + list(site.get("seeds", ())):
            key = (site["country"], host(url))
            if not key[1]:
                raise ValueError(f"{name}: URL has no host: {url}")
            uses.setdefault(key, {}).setdefault(site["use"], set()).add(name)
            all_uses.setdefault(key[1], {}).setdefault(site["use"], set()).add(name)
    for publisher, sites in all_uses.items():
        if len(sites) > 1:
            raise ValueError(f"{publisher} is both silver and eval: {sites}")
    return uses


def fetch_html(url):
    r = common.get(url)
    if r is None or r.status_code != 200:
        return None
    # A redirect to a different publisher must not inherit the candidate's split or licence.
    if host(r.url) != host(url):
        raise ValueError(f"cross-publisher redirect: {url} -> {r.url}")
    return BeautifulSoup(r.content, "html.parser")


def article_links(site):
    seen = []
    for seed in site["seeds"]:
        soup = fetch_html(seed)
        if soup is None:
            continue
        base = soup.find("base", href=True)
        root = urljoin(seed, base["href"]) if base else seed
        for a in soup.find_all("a", href=True):
            url = urljoin(root, a["href"]).split("#")[0]
            if re.search(site["link"], url) and url not in seen:
                seen.append(url)
    return seen


def bounded_source_text(text):
    """Return complete extracted text within the input budget, or reject it."""
    text = common.normalize(text)
    return text if text and len(text) <= common.MAX_CHARS else None


def economy_news_text(soup):
    """Extract a Georgian Economy article only when its title, date, and body are present."""
    carousel = soup.find(id="newsCarousel")
    title = carousel.find_next_sibling("div") if carousel else None
    date = title.find_next_sibling("div") if title else None
    body = date.find_next_sibling("div") if date else None
    if not title or not date or not body or not title.get_text(" ", strip=True):
        return None
    if "font-family:dejavu" not in re.sub(r"\s+", "", body.get("style", "")).lower():
        return None
    try:
        published = datetime.datetime.strptime(date.get_text(" ", strip=True), "%d-%m-%Y").date()
    except ValueError:
        return None
    if published <= datetime.date(1970, 1, 1):
        return None
    text = bounded_source_text(common.block_text(body))
    return text or None


def main(name, limit):
    declared_hosts = source_hosts()
    site = SITES[name]
    common.banner(name, *site["licence"])
    if site["use"] == "silver":
        common.ROOT = common.REPO / "data" / "raw" / "silver"
    pages = site.get("pages") or article_links(site)
    print(f"{name}: {len(pages)} candidate pages", flush=True)

    def fetch(url):
        key = (site["country"], host(url))
        if site["use"] not in declared_hosts.get(key, {}):
            raise ValueError(f"{name}: candidate URL has undeclared publisher: {url}")
        # Georgian slugs lose their letters in a file-safe id, so a hash keeps ids apart.
        digest = hashlib.sha1(url.encode()).hexdigest()[:8]
        doc_id = f"{name}-{common.safe_id(url.split('//', 1)[1])[:80]}-{digest}"
        if common.exists(name, doc_id):
            return None
        soup = fetch_html(url)
        if soup is None:
            return False
        if name == "ge-economy":
            text = economy_news_text(soup)
            if text is None:
                return False
        else:
            text = bounded_source_text(common.content_text(soup))
        if text is None:
            return False
        if site["contact"] == "any":
            wanted = common.has_contact(text) or DE_POSTCODE.search(text)
        else:
            wanted = not site["contact"] or common.has_phone(text)
        if len(text) < MIN_CHARS or not wanted:
            return False
        common.write({"id": doc_id, "source": name, "url": url, "date": TODAY,
                      "country": site["country"], "doc_type": site["doc_type"], "text": text,
                      "licence": site["licence"][0]})
        return True

    common.run(name, pages, fetch, limit)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
