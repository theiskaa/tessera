//! Text that looks like an entity and is not one: all-caps section headings, legal citations,
//! names of laws, programmes, and forms, Title Case label lines, opening hours, acronyms that
//! name no organization, money amounts and order numbers, and code and log lines. Real documents are full of these; without
//! them a detector learns that any capitalized run, and any line with numbers and commas, is
//! a name or an address.
//!
//! Nothing here may name a person, an organization, or a postal address.

use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;

use crate::generate::localized;

/// Section headings in capitals, often ending in a colon.
pub const CAPS_HEADINGS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "SUPPLEMENTARY INFORMATION:",
            "FOR FURTHER INFORMATION CONTACT:",
            "ADDRESSES:",
            "SUMMARY:",
            "DATES:",
            "ACTION:",
            "TERMS AND CONDITIONS",
            "IMPORTANT NOTICE",
            "PAYMENT DETAILS",
            "SYSTEM INFORMATION",
            "SHIPPING INFORMATION",
            "PRIVACY NOTICE",
            "SCOPE OF WORK",
            "EFFECTIVE DATE:",
        ],
    ),
    (
        "DE",
        &[
            "ALLGEMEINE GESCHÄFTSBEDINGUNGEN",
            "IMPRESSUM",
            "WIDERRUFSBELEHRUNG",
            "DATENSCHUTZHINWEIS",
            "LIEFERBEDINGUNGEN",
        ],
    ),
    (
        "GE",
        &["ხელშეკრულების პირობები", "მნიშვნელოვანი ინფორმაცია"],
    ),
    (
        "JP",
        &[
            "【重要】",
            "■お支払いについて",
            "【注意事項】",
            "■配送について",
        ],
    ),
];

/// Laws, regulations, programmes, services, and forms: capitalized names that are never orgs.
pub const LAWS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "Paperwork Reduction Act of 1995",
            "Freedom of Information Act",
            "Regulatory Flexibility Act",
            "Clean Water Act",
            "Endangered Species Act",
            "National Environmental Policy Act",
            "Privacy Act of 1974",
            "Data Protection Act 2018",
            "UK GDPR",
            "Equality Act 2010",
            "Employment Rights Act 1996",
            "Supplemental Nutrition Assistance Program",
            "Housing Choice Voucher Program",
            "Universal Credit",
            "Self Assessment",
            "Making Tax Digital",
            "Form W-9",
            "Form I-9",
            "Standard Form 424",
            "Annual Compliance Report",
            "Annual Burden Estimate",
            "Small Business Grant Programme",
            "Green Skills Programme",
            "General Data Protection Regulation",
        ],
    ),
    (
        "DE",
        &[
            "Datenschutz-Grundverordnung",
            "Bürgerliches Gesetzbuch",
            "Handelsgesetzbuch",
            "Umsatzsteuergesetz",
            "Telemediengesetz",
            "Arbeitszeitgesetz",
            "Elterngeld",
            "Kurzarbeitergeld",
        ],
    ),
    (
        "GE",
        &[
            "საგადასახადო კოდექსი",
            "შრომის კოდექსი",
            "სამოქალაქო კოდექსი",
        ],
    ),
    (
        "JP",
        &[
            "個人情報保護法",
            "電子帳簿保存法",
            "インボイス制度",
            "労働基準法",
            "下請法",
        ],
    ),
];

/// Label lines of forms and notices, in Title Case, with their value.
pub const LABELS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "Respondent Obligation: Voluntary.",
            "Total Annual Cost Burden: none.",
            "Estimated Number of Respondents: 1,250.",
            "Estimated Total Annual Burden Hours: 3,400.",
            "Type of Review: Extension of a currently approved collection.",
            "Courier Deliveries: see the instructions above.",
            "Phone Line: open weekdays.",
            "Method of Collection: Electronic submission.",
            "Payment Terms: Net 30",
            "Delivery Method: Standard Ground",
            "Account Type: Business Premium",
            "Case Status: Awaiting Review",
            "Priority: High",
            "Order Status: Dispatched",
        ],
    ),
    (
        "DE",
        &[
            "Zahlungsart: Überweisung",
            "Lieferart: Standardversand",
            "Bearbeitungsstatus: In Prüfung",
            "Vertragslaufzeit: 24 Monate",
        ],
    ),
    (
        "GE",
        &["გადახდის მეთოდი: საბანკო გადარიცხვა", "სტატუსი: განხილვაში"],
    ),
    (
        "JP",
        &[
            "お支払方法：銀行振込",
            "配送方法：通常便",
            "ステータス：確認中",
        ],
    ),
];

/// Opening hours and service times.
pub const HOURS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "Monday to Friday, 9am to 5pm",
            "Mon–Fri 08:30–17:00",
            "Open weekdays 8am to 6pm, Saturday 9am to 1pm",
            "Lines are open Monday to Thursday, 8.30am to 5pm, Friday 8.30am to 4.30pm",
            "Closed on bank holidays",
            "24 hours a day, 7 days a week",
        ],
    ),
    (
        "DE",
        &[
            "Mo–Fr 8:00–16:30 Uhr",
            "Montag bis Donnerstag 9 bis 17 Uhr, Freitag 9 bis 14 Uhr",
            "Sprechzeiten: Di und Do 10–12 Uhr",
        ],
    ),
    (
        "GE",
        &[
            "ორშაბათი–პარასკევი, 10:00–18:00",
            "სამუშაო საათები: 09:00–17:00",
        ],
    ),
    (
        "JP",
        &[
            "平日 9:00〜17:00",
            "受付時間：10時〜18時（土日祝を除く）",
            "年中無休 24時間受付",
        ],
    ),
];

/// Sentences in Title Case or with acronyms that name no organization.
pub const TITLE_CASE_SENTENCES: &[&str] = &[
    "Please complete the Annual Review Form before the deadline.",
    "The Terms and Conditions apply to every order.",
    "See the Frequently Asked Questions page for details.",
    "Your Monthly Statement is now available in the Customer Portal.",
    "The Quarterly Progress Report is attached.",
    "We follow the Code of Conduct for all suppliers.",
    "The PRA burden estimate covers the time to read the instructions.",
    "Send the signed NDA before the kickoff call.",
    "The SLA covers response times, not resolution times.",
    "Attach the PDF and the CSV export to the ticket.",
    "Your VAT receipt shows the net and gross amounts.",
    "The FAQ and the API reference were updated last week.",
    "Check the ETA on the tracking page.",
    "The Board Meeting Minutes are in the shared folder.",
    "Our Privacy Policy explains how the data is stored.",
    "The Service Level Agreement renews each April.",
    "The Risk Assessment and the Method Statement are both approved.",
    "A FOIA request may take up to twenty working days.",
    "Your PAYE reference is on the payslip.",
    "The KPI dashboard refreshes every morning.",
];

/// Lines of code, configuration, logs, and patches, as mailing lists quote them.
pub const CODE: &[&str] = &[
    "CONFIG_DRM_I915=y",
    "CONFIG_USB_XHCI_HCD=m",
    "# CONFIG_DEBUG_INFO_BTF is not set",
    "[    3.118402] usb 2-1: new SuperSpeed USB device number 2 using xhci_hcd",
    "[   12.004411] EXT4-fs (nvme0n1p2): mounted filesystem with ordered data mode",
    "[    0.000000] Linux version 6.17.0-rc3 (builder@buildhost) #1 SMP PREEMPT_DYNAMIC",
    "[  101.332190] WARNING: CPU: 3 PID: 1187 at mm/page_alloc.c:4410 __alloc_pages+0x2a1/0x330",
    "Call Trace:",
    " <TASK>",
    " dump_stack_lvl+0x5d/0x80",
    " __warn+0x7c/0x130",
    "drivers/net/ethernet/foo/foo_main.c:foo_open()",
    "fs/btrfs/inode.c | 12 ++++++------",
    " 2 files changed, 18 insertions(+), 7 deletions(-)",
    "@@ -120,7 +120,9 @@ static int foo_probe(struct platform_device *pdev)",
    "-\tret = devm_request_irq(dev, irq, foo_irq, 0, \"foo\", priv);",
    "+\tret = devm_request_threaded_irq(dev, irq, NULL, foo_irq,",
    "Fixes: 3f2a9c1e4b7d (\"net: fix race in queue teardown\")",
    "Link: https://lore.kernel.example/r/20260912101500.1234-1-dev@example",
    "Bus 001 Device 003: ID 8087:0026 Bluetooth wireless interface",
    "$ git log --oneline -3",
    "error[E0502]: cannot borrow `buf` as mutable because it is also borrowed as immutable",
    "Traceback (most recent call last):",
    "  File \"setup.py\", line 42, in <module>",
    "ValueError: invalid literal for int() with base 10: 'N/A'",
    "npm ERR! code ERESOLVE",
    "kernel: nvme nvme0: I/O 12 QID 4 timeout, aborting",
    "Hardware name: QEMU Standard PC (Q35 + ICH9, 2009), BIOS 1.16.3-2 04/01/2014",
];

/// Mailing list names, written before a list address.
pub const LISTS: &[&str] = &[
    "linux-usb",
    "dri-devel",
    "netdev",
    "linux-arm-kernel",
    "u-boot",
    "qemu-devel",
    "op-tee",
    "yocto",
];

/// A legal citation with its numbers drawn fresh: `16 U.S.C. 1801 et seq.`, `§ 14 Abs. 2 BGB`.
pub fn citation(country: &str, rng: &mut ChaCha8Rng) -> String {
    let n = |rng: &mut ChaCha8Rng, lo: u32, hi: u32| rng.random_range(lo..=hi);
    if country == "DE" && rng.random_bool(0.6) {
        return match rng.random_range(0..3) {
            0 => format!(
                "§ {} Abs. {} {}",
                n(rng, 1, 400),
                n(rng, 1, 5),
                ["BGB", "HGB", "UStG", "AO"]
                    .choose(rng)
                    .copied()
                    .unwrap_or("BGB")
            ),
            1 => format!(
                "Art. 6 Abs. 1 lit. {} DSGVO",
                ["a", "b", "c", "f"].choose(rng).copied().unwrap_or("f")
            ),
            _ => format!("§§ {} ff. ZPO", n(rng, 1, 900)),
        };
    }
    if country == "JP" && rng.random_bool(0.6) {
        return format!("第{}条第{}項", n(rng, 1, 120), n(rng, 1, 4));
    }
    match rng.random_range(0..6) {
        0 => format!("{} U.S.C. {} et seq.", n(rng, 5, 50), n(rng, 100, 9999)),
        1 => format!(
            "{} CFR {}.{}",
            n(rng, 1, 50),
            n(rng, 1, 999),
            n(rng, 1, 199)
        ),
        2 => format!(
            "{} FR {} ({} {}, {})",
            n(rng, 80, 91),
            n(rng, 1000, 99999),
            ["January", "March", "May", "July", "October"]
                .choose(rng)
                .copied()
                .unwrap_or("May"),
            n(rng, 1, 28),
            n(rng, 2015, 2026)
        ),
        3 => format!(
            "section {}({})({})",
            n(rng, 100, 9999),
            ["a", "b", "c", "d"].choose(rng).copied().unwrap_or("c"),
            n(rng, 1, 9)
        ),
        4 => format!(
            "Regulation (EU) {}/{}",
            n(rng, 2015, 2025),
            n(rng, 100, 2999)
        ),
        _ => format!(
            "paragraph {}.{} of Schedule {}",
            n(rng, 1, 30),
            n(rng, 1, 9),
            n(rng, 1, 12)
        ),
    }
}

/// A directory unit code, written in brackets beside the unit it codes: `MOVE.01`, `ENER.C.2`.
pub fn unit_code(rng: &mut ChaCha8Rng) -> String {
    let head = [
        "MOVE", "ENER", "COMP", "GSC", "SANTE", "TRADE", "CNECT", "HR", "DIGIT", "AGRI",
    ]
    .choose(rng)
    .copied()
    .unwrap_or("HR");
    match rng.random_range(0..3) {
        0 => format!("{head}.{:02}", rng.random_range(1..=9)),
        1 => format!(
            "{head}.{}.{}",
            ["A", "B", "C", "D"].choose(rng).copied().unwrap_or("A"),
            rng.random_range(1..=6)
        ),
        _ => format!("{head}.DTO"),
    }
}

/// A company register number as imprints print it: `HRB 40522`, `Company No. 08123456`.
pub fn register_no(country: &str, rng: &mut ChaCha8Rng) -> String {
    match country {
        "DE" => format!(
            "{} {}",
            ["HRB", "HRA"].choose(rng).copied().unwrap_or("HRB"),
            rng.random_range(1000..=250_000)
        ),
        "GB" => format!("Company No. {:08}", rng.random_range(100_000..=15_999_999)),
        "US" => format!("File No. {}", rng.random_range(1_000_000..=9_999_999)),
        "JP" => format!(
            "法人番号 {}",
            rng.random_range(1_000_000_000_000u64..=9_999_999_999_999)
        ),
        _ => format!("ს/კ {}", rng.random_range(200_000_000..=449_999_999)),
    }
}

/// One negative line for filler blocks and the `neg_*` slots: a heading, a law or form, a
/// label line, opening hours, a citation, or a Title Case sentence.
pub fn line(country: &str, rng: &mut ChaCha8Rng) -> String {
    match rng.random_range(0..7) {
        6 => money(country, true, rng),
        0 => localized(CAPS_HEADINGS, country, 0.4, rng).to_string(),
        1 => localized(LAWS, country, 0.4, rng).to_string(),
        2 => localized(LABELS, country, 0.4, rng).to_string(),
        3 => localized(HOURS, country, 0.4, rng).to_string(),
        4 => citation(country, rng),
        _ => TITLE_CASE_SENTENCES
            .choose(rng)
            .copied()
            .unwrap_or("")
            .to_string(),
    }
}

/// `n` with its thousands separated by `sep`: `4 250`, `4.250`, `4,250`.
fn grouped(n: u64, sep: &str) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push_str(sep);
        }
        out.push(c);
    }
    out
}

/// A money amount written the way `country` writes one, most often in its own currency:
/// `4 250,00 ₾`, `1 240 ლარი`, `1.250,00 €`, `$4,250.00`, `£12.50`, `4,250円`, `３万円`. With
/// `lead`, a word such as `თანხა`, `Betrag`, `Total`, or `合計` often comes before it. Amounts
/// sit beside order numbers, dates, and addresses in invoices and letters, and their digit
/// groups look like house numbers and postcodes.
pub fn money(country: &str, lead: bool, rng: &mut ChaCha8Rng) -> String {
    let n: u64 = match rng.random_range(0..4) {
        0 => rng.random_range(1..100),
        1 => rng.random_range(100..10_000),
        2 => rng.random_range(10_000..1_000_000),
        _ => rng.random_range(1_000_000..50_000_000),
    };
    let cents = rng.random_range(0..100u32);
    // English documents quote other currencies now and then; the others write their own.
    let country = if matches!(country, "US" | "GB") && rng.random_bool(0.15) {
        "US"
    } else {
        country
    };
    let pick = |list: &[&'static str], rng: &mut ChaCha8Rng| -> &'static str {
        list.choose(rng).copied().unwrap_or("")
    };
    let (amount, leads): (String, &[&str]) = match country {
        "GE" => {
            let g = grouped(n, if rng.random_bool(0.6) { " " } else { "" });
            let amount = match rng.random_range(0..7) {
                0 => format!("{g},{cents:02} ₾"),
                1 => format!("{g} ₾"),
                2 => format!("₾{g}"),
                3 => format!("{g} ლარი"),
                4 => format!("{g},{cents:02} ლარი"),
                5 => format!("{g} ლარი და {cents} თეთრი"),
                _ => format!("{g} GEL"),
            };
            let leads: &[&str] = &[
                "თანხა",
                "ღირებულება",
                "ჯამი",
                "ავანსი",
                "გადასახდელი თანხა",
                "ბიუჯეტი",
                "ჯარიმა",
                "ფასი",
                "ხელფასი",
            ];
            (amount, leads)
        }
        "DE" => {
            let g = grouped(n, if rng.random_bool(0.7) { "." } else { "" });
            let amount = match rng.random_range(0..6) {
                0 => format!("{g},{cents:02} €"),
                1 => format!("{g} €"),
                2 => format!("€ {g},{cents:02}"),
                3 => format!("{g},{cents:02} EUR"),
                4 => format!("EUR {g}"),
                _ => format!("{g} Euro"),
            };
            let leads: &[&str] = &[
                "Betrag",
                "Gesamtbetrag",
                "Rechnungsbetrag",
                "Summe",
                "Preis",
                "Kosten",
                "Honorar",
            ];
            (amount, leads)
        }
        "GB" => {
            let g = grouped(n, ",");
            let amount = match rng.random_range(0..4) {
                0 => format!("£{g}.{cents:02}"),
                1 => format!("£{g}"),
                2 => format!("{g} GBP"),
                _ => format!("£{g} (inc. VAT)"),
            };
            let leads: &[&str] = &["Total", "Amount due", "Price", "Fee", "Balance", "Cost"];
            (amount, leads)
        }
        "JP" => {
            let g = grouped(n, ",");
            let amount = match rng.random_range(0..5) {
                0 => format!("{g}円"),
                1 => format!("￥{g}"),
                2 => format!("{g}円（税込）"),
                3 => format!("{}万円", (n / 10_000).max(1)),
                _ => format!("{g} 円"),
            };
            let leads: &[&str] = &["合計", "金額", "請求額", "税込価格", "費用", "予算額"];
            (amount, leads)
        }
        _ => {
            let g = grouped(n, ",");
            let amount = match rng.random_range(0..4) {
                0 => format!("${g}.{cents:02}"),
                1 => format!("${g}"),
                2 => format!("USD {g}"),
                _ => format!("{g} dollars"),
            };
            let leads: &[&str] = &["Total", "Amount due", "Balance", "Fee", "Cost", "Award"];
            (amount, leads)
        }
    };
    if !lead || rng.random_bool(0.4) {
        return amount;
    }
    let colon = if rng.random_bool(0.5) { ":" } else { "" };
    format!("{}{colon} {amount}", pick(leads, rng))
}

/// A date the way `country` writes one: `30.09.2026`, `2026 წლის 30 სექტემბერი`,
/// `30. September 2026`, `2026年9月30日`, `令和8年9月30日`.
pub fn date(country: &str, rng: &mut ChaCha8Rng) -> String {
    let (y, m, d) = (
        rng.random_range(2019..=2027),
        rng.random_range(1..=12usize),
        rng.random_range(1..=28),
    );
    const GE_MONTHS: [&str; 12] = [
        "იანვარი",
        "თებერვალი",
        "მარტი",
        "აპრილი",
        "მაისი",
        "ივნისი",
        "ივლისი",
        "აგვისტო",
        "სექტემბერი",
        "ოქტომბერი",
        "ნოემბერი",
        "დეკემბერი",
    ];
    const DE_MONTHS: [&str; 12] = [
        "Januar",
        "Februar",
        "März",
        "April",
        "Mai",
        "Juni",
        "Juli",
        "August",
        "September",
        "Oktober",
        "November",
        "Dezember",
    ];
    match (country, rng.random_range(0..3)) {
        ("GE", 0) => format!("{y} წლის {d} {}", GE_MONTHS[m - 1]),
        ("DE", 0) => format!("{d}. {} {y}", DE_MONTHS[m - 1]),
        ("JP", 0) => format!("{y}年{m}月{d}日"),
        ("JP", 1) => format!("令和{}年{m}月{d}日", y - 2018),
        ("JP", _) => format!("{y}/{m:02}/{d:02}"),
        _ => format!("{d:02}.{m:02}.{y}"),
    }
}

/// An order, invoice, or case number the way `country` writes one: `№7781`, `INV-2026-0412`,
/// `Nr. 20417`, `No. 2026-118`, `第1204号`.
pub fn order(country: &str, rng: &mut ChaCha8Rng) -> String {
    let n = rng.random_range(10..99_999);
    let year = rng.random_range(2019..=2027);
    match (country, rng.random_range(0..4)) {
        ("GE", 0 | 1) => format!("№{n}"),
        ("GE", 2) => format!("№ {n}"),
        ("GE", _) => format!("{year}/{n:04}"),
        ("DE", 0 | 1) => format!("Nr. {n}"),
        ("DE", 2) => format!("RE-{year}-{n:04}"),
        ("DE", _) => format!("{year}-{n:05}"),
        ("JP", 0) => format!("第{n}号"),
        ("JP", 1) => format!("No. {year}-{n}"),
        ("JP", _) => format!("{year}-{n:05}"),
        (_, 0) => format!("INV-{year}-{n:04}"),
        (_, 1) => format!("#{n}"),
        (_, 2) => format!("{year}/{n:04}"),
        _ => format!("PO {n}"),
    }
}

/// Two to five consecutive code, config, or log lines.
pub fn code_block(rng: &mut ChaCha8Rng) -> String {
    let start = rng.random_range(0..CODE.len());
    let n = rng.random_range(2..=5);
    (0..n)
        .map(|i| CODE[(start + i) % CODE.len()])
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn money_is_written_in_each_country_s_currency() {
        assert_eq!(grouped(4250, " "), "4 250");
        assert_eq!(grouped(1234567, "."), "1.234.567");
        assert_eq!(grouped(999, ","), "999");
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let marks = [
            ("GE", &["₾", "ლარ", "GEL", "$", "USD", "dollars"][..]),
            ("DE", &["€", "EUR", "Euro", "$", "USD", "dollars"][..]),
            ("JP", &["円", "￥", "$", "USD", "dollars"][..]),
        ];
        for (country, marks) in marks {
            let mut native = 0;
            for _ in 0..200 {
                let m = money(country, true, &mut rng);
                assert!(marks.iter().any(|k| m.contains(k)), "{country}: {m}");
                native +=
                    usize::from(!m.contains('$') && !m.contains("USD") && !m.contains("dollars"));
            }
            assert!(native > 150, "{country}: {native}");
        }
        for country in ["US", "GB", "DE", "GE", "JP"] {
            assert!(!order(country, &mut rng).is_empty());
        }
    }

    #[test]
    fn negative_lines_are_never_empty_and_hold_no_slot_braces() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for country in ["US", "GB", "DE", "GE", "JP"] {
            for _ in 0..200 {
                let l = line(country, &mut rng);
                assert!(!l.is_empty());
                assert!(!l.contains(['{', '}']), "{l}");
            }
        }
        let block = code_block(&mut rng);
        assert!(block.lines().count() >= 2);
    }
}
