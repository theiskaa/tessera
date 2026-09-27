import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

import fetch_sites
from fetch_sites import bounded_source_text, economy_news_text


def page(title, date, body):
    return BeautifulSoup(
        f'<div class="sidebar">{"სერვისები " * 100}</div>'
        '<div id="newsCarousel"><div class="carousel-inner"></div></div>'
        f'<div>{title}</div><div>{date}</div>'
        f'<div style="font-family:dejavu; text-align:justify">{body}</div>'
        '<div class="footer">დაგვიკავშირდით 1403</div>',
        "html.parser",
    )


class GeorgianEconomyArticleTests(unittest.TestCase):
    def test_valid_article_uses_body_only(self):
        soup = page("მარიამ ქვრივიშვილი შეხვედრას დაესწრო", "24-09-2026",
                    "<p>პირველი აბზაცი.</p><p>მეორე აბზაცი.</p>")
        self.assertEqual(economy_news_text(soup), "პირველი აბზაცი.\n\nმეორე აბზაცი.")

    def test_oversize_article_is_rejected(self):
        paragraphs = "".join(f"<p>შინაარსი {i}: " + "აბგ" * 30 + "</p>" for i in range(100))
        self.assertIsNone(economy_news_text(page("Headline", "24-09-2026", paragraphs)))
        self.assertIsNone(bounded_source_text("Start\n\n" + "x" * 6001 + "\n\nEnd"))
        self.assertEqual(bounded_source_text(" First\n\nSecond "), "First\n\nSecond")

    def test_site_shell_is_not_a_news_article(self):
        self.assertIsNone(economy_news_text(page("", "01-01-1970", "")))
        self.assertIsNone(economy_news_text(page("", "24-09-2026", "<p>Body</p>")))
        self.assertIsNone(economy_news_text(page("Headline", "24-09-2026", "")))

    def test_invalid_date_or_missing_article_container_is_rejected(self):
        self.assertIsNone(economy_news_text(page("Headline", "not a date", "Body")))
        self.assertIsNone(economy_news_text(page("Headline", "01-01-1970", "Body")))
        self.assertIsNone(economy_news_text(BeautifulSoup("<div>Body</div>", "html.parser")))

    def test_sidebar_cannot_become_article_body(self):
        soup = BeautifulSoup(
            '<div id="newsCarousel"></div><div>Headline</div><div>24-09-2026</div>'
            f'<div class="sidebar">{"სერვისები " * 100}</div>', "html.parser")
        self.assertIsNone(economy_news_text(soup))


class StandardPageTests(unittest.TestCase):
    def test_oversize_page_is_not_written_but_bounded_page_is(self):
        url = "https://example.org/contact"
        site = {"test": {"use": "eval", "country": "GB", "doc_type": "contact page",
                         "licence": ("test", "test"), "pages": [url], "contact": False}}
        results = []

        def one_page(_name, pages, fetch, _limit):
            results.append(fetch(pages[0]))

        with patch.object(fetch_sites, "SITES", site), \
             patch.object(fetch_sites, "fetch_html") as fetch_html, \
             patch.object(fetch_sites.common, "exists", return_value=False), \
             patch.object(fetch_sites.common, "write") as write, \
             patch.object(fetch_sites.common, "banner"), \
             patch.object(fetch_sites.common, "run", side_effect=one_page):
            fetch_html.return_value = BeautifulSoup(
                "<main>" + "".join(f"<p>{letter * 2500}</p>" for letter in "ABC") + "</main>",
                "html.parser")
            fetch_sites.main("test", 1)
            self.assertIs(results[-1], False)
            write.assert_not_called()

            fetch_html.return_value = BeautifulSoup(
                "<main>" + "".join(f"<p>{letter * 200}</p>" for letter in "ABC") + "</main>",
                "html.parser")
            fetch_sites.main("test", 1)
            self.assertIs(results[-1], True)
            self.assertEqual(write.call_args.args[0]["text"],
                             "\n\n".join(letter * 200 for letter in "ABC"))


if __name__ == "__main__":
    unittest.main()
