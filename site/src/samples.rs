//! The demo documents. Everything highlighted in them is found by `extract_contacts` at run
//! time; the contact details are reserved for fiction or documentation.

/// One demo document.
pub(crate) struct Sample {
    pub(crate) name: &'static str,
    pub(crate) file: &'static str,
    /// The region the document is written in. The demo leaves the library to infer it; the
    /// tests check that inference finds what this hint finds.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "only the tests read the hint; the demo infers it")
    )]
    pub(crate) country_hint: &'static str,
    pub(crate) text: &'static str,
}

pub(crate) const SAMPLES: &[Sample] = &[
    Sample {
        name: "invoice",
        file: "invoice-2026-00931.txt",
        country_hint: "US",
        text: r#"INVOICE #2026-00931
Date: September 18, 2026 · Due: October 18, 2026 · PO 55-1920-A

Bill to:
Meridian Health Partners
Attn: Grace Liu, Accounts Payable
1200 Market Street, Suite 400
Philadelphia, PA 19107

From:
Blue Ridge Supply Co
88 Commerce Drive, Asheville, NC 28801
(828) 555-0142 · billing@blueridge-supply.example

Item                     Qty   Unit      Total
Nitrile gloves, case      40   $38.50    $1,540.00
Sharps containers         12   $14.25      $171.00
Shipping (UPS 1Z 999 AA1 01 2345 6784)       $86.40
Total due                                 $1,797.40

Questions? Call Daniel Price on (828) 555-0187.
"#,
    },
    Sample {
        name: "signature",
        file: "project-update.txt",
        country_hint: "US",
        text: r#"Hi Sam,

The revised drawings are ready. Let me know if you need a printed copy before Thursday's meeting.

Jordan Avery, Project Coordinator
Acme Corporation
500 Main St, Springfield, IL 62701
+1 202 555 0199
jordan@acme.example
"#,
    },
    Sample {
        name: "directory",
        file: "team-directory.txt",
        country_hint: "US",
        text: r#"TEAM DIRECTORY

Customer care
Avery Parker
Harbor Desk Studio
400 Broad St, Seattle, WA 98109
+1 206 555 0148
support@harbordesk.example

Appointments
Casey Morgan
Maple Line Clinic
120 Main St, Austin, TX 78701
+1 512 555 0162
bookings@mapleline.example

Office hours: Monday–Friday, 9:00–17:00
"#,
    },
    Sample {
        name: "paragraph",
        file: "delivery-note.txt",
        country_hint: "US",
        text: r#"For Thursday's delivery, ask Riley Morgan at Meadow Supply Co. The loading desk is at 2100 N Lamar Blvd, Suite 210, Dallas, TX 75202. Contact Riley on +1 214 555 0173 or riley@meadow-supply.example if the gate is closed.

Reference: PKG-2048. Please arrive between 10:00 and 12:00 and keep the receipt with the parcel.
"#,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use tessera::{Config, Kind, Query, Tessera};

    /// The library with the rules only, which need no bundle.
    fn rules() -> Tessera {
        Tessera::load(
            &[],
            Config {
                kinds: Kind::Email | Kind::Phone,
                expected_checksum: None,
            },
        )
        .unwrap()
    }

    fn rules_output(sample: &Sample) -> Vec<(String, String)> {
        let hint = [sample.country_hint];
        let query = Query {
            country_hint: &hint,
            ..Query::default()
        };
        rules()
            .detect(sample.text, &query)
            .unwrap()
            .iter()
            .map(|e| (e.kind.as_str().to_string(), e.text(sample.text).to_string()))
            .collect()
    }

    /// The rules find exactly these in each sample, and none of its order, ticket, invoice,
    /// tracking, tax, or IBAN numbers.
    #[test]
    fn the_rules_find_exactly_the_contact_details() {
        let want: [&[(&str, &str)]; 4] = [
            &[
                ("phone", "(828) 555-0142"),
                ("email", "billing@blueridge-supply.example"),
                ("phone", "(828) 555-0187"),
            ],
            &[
                ("phone", "+1 202 555 0199"),
                ("email", "jordan@acme.example"),
            ],
            &[
                ("phone", "+1 206 555 0148"),
                ("email", "support@harbordesk.example"),
                ("phone", "+1 512 555 0162"),
                ("email", "bookings@mapleline.example"),
            ],
            &[
                ("phone", "+1 214 555 0173"),
                ("email", "riley@meadow-supply.example"),
            ],
        ];
        assert_eq!(SAMPLES.len(), want.len());
        for (sample, want) in SAMPLES.iter().zip(want) {
            let want: Vec<(String, String)> = want
                .iter()
                .map(|(k, t)| (k.to_string(), t.to_string()))
                .collect();
            assert_eq!(rules_output(sample), want, "{}", sample.name);
        }
    }

    /// The demo reads regions from each document, so no sample may need its hint.
    #[test]
    fn inferred_regions_find_what_the_hint_finds() {
        let rules = rules();
        for sample in SAMPLES {
            let found = |hint: &[&str]| -> Vec<(usize, usize, Option<String>)> {
                let query = Query {
                    country_hint: hint,
                    ..Query::default()
                };
                rules
                    .detect(sample.text, &query)
                    .unwrap()
                    .into_iter()
                    .map(|e| (e.start, e.end, e.normalized))
                    .collect()
            };
            assert_eq!(found(&[]), found(&[sample.country_hint]), "{}", sample.name);
        }
    }

    #[test]
    fn names_and_files_are_distinct() {
        for (i, a) in SAMPLES.iter().enumerate() {
            for b in &SAMPLES[i + 1..] {
                assert!(a.name != b.name && a.file != b.file, "{}", a.name);
            }
        }
    }
}
