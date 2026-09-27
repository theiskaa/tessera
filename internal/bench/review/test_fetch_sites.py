import unittest

from bs4 import BeautifulSoup

from fetch_sites import economy_news_text


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


if __name__ == "__main__":
    unittest.main()
