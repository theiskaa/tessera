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
    },
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
        name: "letter",
        file: "angebot-2026-1187.txt",
        country_hint: "DE",
        text: r#"Brandt & Söhne GmbH
Friedrichstraße 88
10117 Berlin
Tel. 030 23125 456 · info@brandt-soehne.example

Weber Logistik GmbH
z. Hd. Herrn Jonas Weber
Hafenstraße 12
20457 Hamburg

Berlin, 12. Oktober 2026

Angebot Nr. 2026-1187 · Kundennummer 40 118 · USt-IdNr. DE 999 999 999

Sehr geehrter Herr Weber,

vielen Dank für Ihre Anfrage vom 8. Oktober. Für die Lieferung von 24 Paletten berechnen wir 3.980,00 € zzgl. MwSt., zahlbar innerhalb von 14 Tagen auf IBAN DE89 3704 0044 0532 0130 00.

Mit freundlichen Grüßen

Katrin Vogel
Vertrieb
Durchwahl 030 23125 457
"#,
    },
    Sample {
        name: "georgian",
        file: "inv-2026-0587.eml",
        country_hint: "GE",
        text: r#"თემა: ინვოისი INV-2026-0587, გადახდის ვადა 30.09.2026

გამარჯობა,

გიგზავნით ინვოისს სექტემბრის მიწოდებისთვის (შეკვეთა №7781, თანხა 4 250,00 ₾). ანგარიში: GE29NB0000000101904917.

მადლობა, შეხვედრამდე ორშაბათს.

ნინო ბერიძე
შპს კავკასიის ტვირთი
რუსთაველის გამზირი 14, თბილისი 0108
+995 32 212 3456
nino@kavkaz-freight.example
"#,
    },
    Sample {
        name: "japanese",
        file: "mitsumori.eml",
        country_hint: "JP",
        text: r#"件名：お見積書（No. Q-2026-0915）送付の件

株式会社みどり商事
営業部 鈴木一郎 様

いつもお世話になっております。
ご依頼いただいた商品の見積書を添付いたします。合計金額は 128,700円（税込）、有効期限は2026年10月31日です。

ご不明な点がございましたら、お気軽にご連絡ください。
よろしくお願いいたします。

山田太郎
株式会社サクラ物流
〒150-0002 東京都渋谷区渋谷2丁目21-1
taro.yamada@sakura-logistics.example
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
        let want: [&[(&str, &str)]; 6] = [
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
            &[
                ("phone", "(828) 555-0142"),
                ("email", "billing@blueridge-supply.example"),
                ("phone", "(828) 555-0187"),
            ],
            &[
                ("phone", "030 23125 456"),
                ("email", "info@brandt-soehne.example"),
                ("phone", "030 23125 457"),
            ],
            &[
                ("phone", "+995 32 212 3456"),
                ("email", "nino@kavkaz-freight.example"),
            ],
            &[("email", "taro.yamada@sakura-logistics.example")],
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
