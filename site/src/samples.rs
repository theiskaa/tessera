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

pub(crate) const SAMPLES: &[Sample] = &[Sample {
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
}];

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
        let want: [&[(&str, &str)]; 1] = [&[
            ("phone", "(828) 555-0142"),
            ("email", "billing@blueridge-supply.example"),
            ("phone", "(828) 555-0187"),
        ]];
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
