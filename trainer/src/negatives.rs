//! US document text that resembles entities but must remain unlabelled.

use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;

use crate::generate::localized;

pub const CAPS_HEADINGS: &[(&str, &[&str])] = &[(
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
)];

pub const LAWS: &[(&str, &[&str])] = &[(
    "en",
    &[
        "Paperwork Reduction Act of 1995",
        "Freedom of Information Act",
        "Regulatory Flexibility Act",
        "Clean Water Act",
        "Endangered Species Act",
        "National Environmental Policy Act",
        "Privacy Act of 1974",
        "Supplemental Nutrition Assistance Program",
        "Housing Choice Voucher Program",
        "Form W-9",
        "Form I-9",
        "Standard Form 424",
        "Annual Compliance Report",
        "Annual Burden Estimate",
    ],
)];

pub const SERVICES: &[(&str, &[&str])] = &[(
    "en",
    &[
        "Social Security",
        "Medicare",
        "Medicaid",
        "SNAP",
        "FAFSA",
        "EIN",
        "TTY",
        "Help Desk",
        "Customer Service",
        "Direct Deposit",
        "E-Verify",
    ],
)];

pub const LABELS: &[(&str, &[&str])] = &[(
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
)];

pub const HOURS: &[(&str, &[&str])] = &[(
    "en",
    &[
        "Monday to Friday, 9am to 5pm",
        "Mon–Fri 08:30–17:00",
        "Open weekdays 8am to 6pm, Saturday 9am to 1pm",
        "Phone lines are open Monday through Friday, 8:30 a.m. to 5:00 p.m.",
        "Closed on federal holidays",
        "24 hours a day, 7 days a week",
    ],
)];

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
    "Your sales tax receipt shows the subtotal and total.",
    "The FAQ and the API reference were updated last week.",
    "Check the ETA on the tracking page.",
    "The Board Meeting Minutes are in the shared folder.",
    "Our Privacy Policy explains how the data is stored.",
    "The Service Level Agreement renews each April.",
    "The Risk Assessment and the Method Statement are both approved.",
    "A FOIA request may take up to twenty working days.",
    "Your payroll reference is on the pay stub.",
    "The KPI dashboard refreshes every morning.",
];

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

/// A US legal or regulatory citation.
pub fn citation(_country: &str, rng: &mut ChaCha8Rng) -> String {
    let n = |rng: &mut ChaCha8Rng, lo: u32, hi: u32| rng.random_range(lo..=hi);
    match rng.random_range(0..4) {
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
        _ => format!(
            "section {}({})({})",
            n(rng, 100, 9999),
            ["a", "b", "c", "d"].choose(rng).copied().unwrap_or("c"),
            n(rng, 1, 9)
        ),
    }
}

/// A short office code next to a named unit.
pub fn unit_code(rng: &mut ChaCha8Rng) -> String {
    let head = ["OPS", "ENG", "FIN", "HR", "IT"]
        .choose(rng)
        .copied()
        .unwrap_or("OPS");
    format!("{head}.{:02}", rng.random_range(1..=99))
}

/// A state corporation filing number.
pub fn register_no(_country: &str, rng: &mut ChaCha8Rng) -> String {
    format!("File No. {}", rng.random_range(1_000_000..=9_999_999))
}

/// One unlabeled line for synthetic filler.
pub fn line(country: &str, rng: &mut ChaCha8Rng) -> String {
    match rng.random_range(0..8) {
        7 => localized(SERVICES, country, 0.5, rng).to_string(),
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

fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A US dollar amount, optionally with a label.
pub fn money(_country: &str, lead: bool, rng: &mut ChaCha8Rng) -> String {
    let n = rng.random_range(1..50_000_000u64);
    let cents = rng.random_range(0..100u32);
    let amount = match rng.random_range(0..4) {
        0 => format!("${}.{:02}", grouped(n), cents),
        1 => format!("${}", grouped(n)),
        2 => format!("USD {}", grouped(n)),
        _ => format!("{} dollars", grouped(n)),
    };
    if !lead || rng.random_bool(0.4) {
        return amount;
    }
    let label = ["Total", "Amount due", "Balance", "Fee", "Cost", "Award"]
        .choose(rng)
        .copied()
        .unwrap_or("Total");
    let colon = if rng.random_bool(0.5) { ":" } else { "" };
    format!("{label}{colon} {amount}")
}

/// A US month-first date.
pub fn date(_country: &str, rng: &mut ChaCha8Rng) -> String {
    let year = rng.random_range(2019..=2027);
    let month = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    .choose(rng)
    .copied()
    .unwrap_or("January");
    let day = rng.random_range(1..=28);
    format!("{month} {day}, {year}")
}

/// A US invoice, order, or case reference.
pub fn order(_country: &str, rng: &mut ChaCha8Rng) -> String {
    let number = rng.random_range(10..99_999);
    let year = rng.random_range(2019..=2027);
    match rng.random_range(0..4) {
        0 => format!("INV-{year}-{number:04}"),
        1 => format!("#{number}"),
        2 => format!("{year}/{number:04}"),
        _ => format!("PO {number}"),
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
    fn us_negative_lines_are_nonempty_and_not_template_slots() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for _ in 0..500 {
            let line = line("US", &mut rng);
            assert!(!line.is_empty());
            assert!(!line.contains(['{', '}']), "{line}");
            assert!(money("US", false, &mut rng).contains(['$', 'U', 'd']));
            assert!(!order("US", &mut rng).is_empty());
        }
        assert!(code_block(&mut rng).lines().count() >= 2);
    }
}
