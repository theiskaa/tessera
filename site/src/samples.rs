//! The demo documents. Emails and phones in them are found by `detect` at run time; each
//! address line is given here as a byte range and split by `parse_address` at run time.

/// One demo document.
pub(crate) struct Sample {
    pub(crate) name: &'static str,
    pub(crate) file: &'static str,
    /// Region for phone numbers written without a country code.
    pub(crate) country_hint: &'static str,
    pub(crate) text: &'static str,
    /// Byte ranges of `text` holding one address each, in document order.
    pub(crate) addresses: &'static [(usize, usize)],
}

pub(crate) const SAMPLES: &[Sample] = &[
    Sample {
        name: "email",
        file: "support.eml",
        country_hint: "GB",
        text: r#"Subject: Re: Order #48213-B not delivered (ticket 2026-0917-4471)

Hi Oliver,

The two pallets from order #48213-B (invoice INV-2026-00871, £3,418.60) were due on 12 September, 08:00–12:00. Tracking 1Z 999 AA1 01 2345 6784 has said "out for delivery" since Monday. Refund details if needed: IBAN GB29 NWBK 6016 1331 9268 19.

Please send the new slot to our warehouse manager:

Priya Raman
Harbourside Supplies Ltd
Unit 7, 42 Dock Road, Liverpool L3 4BN, UK
Tel +44 151 496 0147 · Mobile +44 7700 900123
priya.raman@harbourside.example

The gate code is 4471# and the bay closes at 16:30 sharp. If nobody answers, try me on 020 7946 0321.

Best regards,
Daniel Okafor
Operations, Harbourside Supplies Ltd

> On 15 Sep 2026, at 09:12, Oliver Grant <oliver.grant@northwind.example> wrote:
> Apologies for the delay. Our depot at 3rd Floor, 1 Canada Square, London E14 5AB has re-booked the van for Thursday 18 September, 07:30-09:30.
"#,
        addresses: &[(418, 460), (831, 873)],
    },
    Sample {
        name: "thread",
        file: "thread.eml",
        country_hint: "GB",
        text: r#"Hi Tamar,

Please send the invoice to our Munich office, Musterfirma GmbH, Leopoldstraße 8, 80802 München.

Best,
Oliver Grant
Northwind Freight Ltd
3rd Floor, 1 Canada Square, London E14 5AB
+44 20 7946 0321
oliver.grant@northwind.example

> On 12 Sep, Tamar Kiknadze wrote:
> Our warehouse is at Aghmashenebeli Avenue 12, Tbilisi 0131.
> Call me on +995 32 212 3456 if the truck is late.
> tamar@kavkaz-freight.example
"#,
        addresses: &[(75, 107), (151, 193), (300, 338)],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use tessera::{Config, Kind, Query, Tessera};

    #[test]
    fn address_lines_are_ordered_and_char_aligned() {
        for sample in SAMPLES {
            let mut cursor = 0;
            for &(start, end) in sample.addresses {
                assert!(
                    start >= cursor && start < end,
                    "{}: {start}..{end}",
                    sample.name
                );
                assert!(
                    sample.text.get(start..end).is_some(),
                    "{}: {start}..{end}",
                    sample.name
                );
                cursor = end;
            }
        }
    }

    fn detect(sample: &Sample) -> Vec<(String, String)> {
        let rules = Tessera::load(
            &[],
            Config {
                kinds: Kind::Email | Kind::Phone,
                expected_checksum: None,
            },
        )
        .unwrap();
        let hint = [sample.country_hint];
        let query = Query {
            country_hint: &hint,
            ..Query::default()
        };
        rules
            .detect(sample.text, &query)
            .unwrap()
            .iter()
            .map(|e| (e.kind.as_str().to_string(), e.text(sample.text).to_string()))
            .collect()
    }

    /// Each sample shows exactly these, and none of its order, ticket, invoice, tracking, or
    /// IBAN numbers.
    #[test]
    fn detect_finds_exactly_the_contact_details() {
        let want: [&[(&str, &str)]; 2] = [
            &[
                ("phone", "+44 151 496 0147"),
                ("phone", "+44 7700 900123"),
                ("email", "priya.raman@harbourside.example"),
                ("phone", "020 7946 0321"),
                ("email", "oliver.grant@northwind.example"),
            ],
            &[
                ("phone", "+44 20 7946 0321"),
                ("email", "oliver.grant@northwind.example"),
                ("phone", "+995 32 212 3456"),
                ("email", "tamar@kavkaz-freight.example"),
            ],
        ];
        assert_eq!(SAMPLES.len(), want.len());
        for (sample, want) in SAMPLES.iter().zip(want) {
            let want: Vec<(String, String)> = want
                .iter()
                .map(|(k, t)| (k.to_string(), t.to_string()))
                .collect();
            assert_eq!(detect(sample), want, "{}", sample.name);
        }
    }

    #[test]
    fn detected_entities_do_not_overlap_address_lines() {
        for sample in SAMPLES {
            let rules = Tessera::load(
                &[],
                Config {
                    kinds: Kind::Email | Kind::Phone,
                    expected_checksum: None,
                },
            )
            .unwrap();
            let hint = [sample.country_hint];
            let query = Query {
                country_hint: &hint,
                ..Query::default()
            };
            for e in rules.detect(sample.text, &query).unwrap() {
                for &(start, end) in sample.addresses {
                    assert!(
                        e.end <= start || e.start >= end,
                        "{}: {:?} overlaps {start}..{end}",
                        sample.name,
                        e.text(sample.text)
                    );
                }
            }
        }
    }
}
