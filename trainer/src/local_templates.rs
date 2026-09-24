//! Templates written for particular countries: sentences in the country's language around
//! the entities, the page layouts of its public bodies, and the name shapes its documents use.
//! Georgian names and bodies take case endings (`{person_native:erg#1}`), Japanese ones are
//! glued into the text, German ones follow articles, and English news repeats surnames and
//! acronyms without their definition.

use crate::generate::{Category, Family};
use Category::{Both, NoAddressWithPerson as NoAddr, NoPersonWithAddress as NoPers, Nothing};
use Family::*;

type Local = (Family, u32, Category, &'static [&'static str], &'static str);

const GE: &[&str] = &["GE"];
const JP: &[&str] = &["JP"];
const DE: &[&str] = &["DE"];
const GB: &[&str] = &["GB"];
const US: &[&str] = &["US"];
const NEWS: &[&str] = &["GE", "GB", "US"];

/// The country templates as `(family, id, category, countries, text)`; `templates::all` turns
/// them into `Template`s beside the shared ones.
pub const TEMPLATES: &[Local] = &[
    (
        Prose,
        40,
        NoAddr,
        GE,
        "{sentence_local} {org_gov:gen#1} {title_local:erg} {person_native:erg#2} {org:com#3} შეხვედრა გამართა. {sentence_local}",
    ),
    (
        Prose,
        41,
        NoAddr,
        GE,
        "{person_native#1}, {org_gov:gen#2} {title_local}, აცხადებს, რომ {org#3} პროექტს წლის ბოლომდე დაასრულებს. {sentence_local}",
    ),
    (
        Prose,
        42,
        NoAddr,
        GE,
        "{org_gov:erg#1} {org:com#2} ერთად მემორანდუმი გააფორმა. დოკუმენტს ხელი {person_native:erg#3} და {person_native:erg#4} მოაწერეს. {sentence_local}",
    ),
    (
        Prose,
        43,
        NoAddr,
        GE,
        "{sentence_local} {org_gov:gen#1} ინფორმაციით, {person_native#2} {org:dat#3} ხელმძღვანელობს. {person_last:erg#2} აღნიშნა, რომ {sentence_local}",
    ),
    (
        Prose,
        44,
        NoAddr,
        GE,
        "{org_univ:loc#1} საჯარო ლექცია გაიმართა. ლექცია {person_native:erg#2} წაიკითხა, ღონისძიებას {org_univ:gen#1} სტუდენტები და {org_chain:gen#3} წარმომადგენლები დაესწრნენ.",
    ),
    (
        Prose,
        45,
        NoPers,
        GE,
        "{org_chain:gen#1} ინფორმაციით, {sentence_local}\n\nმისამართი: {address#2}\nტელ: {phone#3}",
    ),
    (
        Prose,
        46,
        NoAddr,
        GE,
        "{sentence_local}\n\n{person_native#1} - {org_univ:gen#2} {title_local}\n{person_native#3} - {org_univ:gen#2} {title_local}\n{person_native#4} - {org_chain#5}",
    ),
    (
        Prose,
        47,
        NoAddr,
        GE,
        "{person_native:gen#1} თქმით, {org_party#2} {org_gov:com#3} თანამშრომლობას გააგრძელებს. {person_last:erg#1} ასევე აღნიშნა, რომ {sentence_local}",
    ),
    (
        Prose,
        49,
        NoAddr,
        GE,
        "{org_gov:erg#1} {person_native:dat#2} მადლობა გადაუხადა. {sentence_local} შეხვედრას {org_chain:gen#3} წარმომადგენლებიც ესწრებოდნენ.",
    ),
    (
        Prose,
        48,
        Both,
        GE,
        "{sentence_local}\n\nანგარიშ-ფაქტურა {neg_order}, თარიღი {neg_date}. გადასახდელია {neg_price}, ხოლო ავანსად უკვე ჩაირიცხა {neg_price}.\n\n{person_native#1}\n{org#2}\n{address#3}\n{phone#1}",
    ),
    (
        Signature,
        240,
        Both,
        GE,
        "{person_native#1}\n{title_local}\n{org_chain#2}\n{address#3}\nტელ: {phone#1}\nელ-ფოსტა: {email#1}",
    ),
    (
        Letterhead,
        340,
        NoPers,
        GE,
        "{org_gov#1}\n\n{sentence_local} {sentence_local}\n\nმისამართი: {address_ml#2}\nცხელი ხაზი: {phone#3}\nელ-ფოსტა: {email#3}",
    ),
    (
        Prose,
        50,
        NoAddr,
        NEWS,
        "{sentence} {title} {person#1} said on {date} that the {org_gov#2} ({org_acr#2}) would review the decision. {person_last#1} added that the {org_acr#2} had received no complaint. {sentence}",
    ),
    (
        Prose,
        51,
        NoAddr,
        NEWS,
        "{org_party#1} lawmaker {person#2} criticized the {org_gov#3} on {date}. {person_last:poss#2} remarks came after {org_acr#4} officials met {person#5}. {sentence}",
    ),
    (
        Prose,
        52,
        NoAddr,
        NEWS,
        "Also Read:\n\n{date} – {org_acr#1} Chief {person_last#2} Steps Down\n\n{date} – {org_gov#3} Approves New Rules\n\nTags\n\n{org_acr#1} {person#2} {org_gov#3} {neg_place}",
    ),
    (
        Prose,
        53,
        NoPers,
        &["GB", "US"],
        "Contact {org_acr#1} if you cannot find the answer online. {org_acr:poss#1} helpline opening hours are {neg_hours}. Write to {org_acr#1} at:\n\n{org_gov#1}\n{address_ml#2}",
    ),
    (
        Prose,
        54,
        NoAddr,
        NEWS,
        "The {org_gov#1} ({org_acr#1}) published new guidance on {date}. Under the {neg_law}, {org_acr#1} must respond within twenty days. {org_acr:poss#1} decision was welcomed by {person#2}, {title} at {org#3}.",
    ),
    (
        Prose,
        59,
        NoAddr,
        NEWS,
        "{person#1}, the {title} of the {org_gov#2}, told reporters that {org_acr#2} staff would meet {org_party#3} members. {person_last#1} did not comment further.",
    ),
    (
        Prose,
        60,
        NoAddr,
        JP,
        "{org_gov#1}は、{date}に{org_council#2}を開催します。\n\n{sentence_local}\n\n【問合せ先】\n{org_chain#3}\n担当：{person_last#4}{title_local}、{person_last#5}{title_local}\n電話：{phone#6}（直通）",
    ),
    (
        Prose,
        61,
        NoAddr,
        JP,
        "{sentence_local}{org_gov#1}は、{org#2}と連携し、取組を進めています。{person_native#3}{title_local}は、{sentence_local}",
    ),
    (
        Prose,
        62,
        NoPers,
        JP,
        "【日時】{date}\n【場所】{address#1}\n【主催】{org_gov#2}、{org#3}\n\n{sentence_local}",
    ),
    (
        Prose,
        63,
        NoAddr,
        JP,
        "{org_council#1}の会長に{person_native#2}氏が就任しました。{sentence_local}\n\n連絡先\n\n{org_chain#3}\n\n代表\n\n{phone#4}\n\n直通\n\n{phone#5}\n\n担当\n\n{person_native#6}",
    ),
    (
        Prose,
        64,
        Nothing,
        JP,
        "{place_local}\n\n{place_local}\n\n0.06 ha\n\n{place_local}\n\n{place_local}\n\n0.12 ha\n\n{sentence_local}",
    ),
    (
        Prose,
        65,
        NoAddr,
        JP,
        "担当者：{org_unit#1}\n担当：{person_last#2}、{person_last#3}\n代表：{phone#4}（内線4975）\nダイヤルイン：{phone#5}",
    ),
    (
        Prose,
        66,
        NoAddr,
        JP,
        "平成16年{org_gov#1}告示第32号に基づき、{org_gov#1}は{sentence_local}\n\n{org_unit#2}\n担当：{person_last#3}係長",
    ),
    (
        Prose,
        69,
        NoAddr,
        JP,
        "{org_gov#1}は、{org_council#2}に諮問しました。\n\n問合せ先\n{org_chain#3}\n担当：{person_last#4}{title_local}\n電話：{phone#5}",
    ),
    (
        Prose,
        67,
        Both,
        JP,
        "{org#1}御中\n\n請求書番号 {neg_order}\n請求金額 {neg_price}\nお支払期限 {neg_date}\n\n{org#2}\n〒{address#3}\n担当：{person_last#4}",
    ),
    (
        Letterhead,
        360,
        Both,
        JP,
        "{org#1}\n〒{address#2}\nTEL {phone#3}\n代表取締役 {person_native#4}",
    ),
    (
        Letterhead,
        370,
        NoAddr,
        DE,
        "Impressum\n\n{role_heading}\n{person_list}\n\n{role_heading}\n{person_list}\n\n{sentence_local}",
    ),
    (
        Letterhead,
        371,
        Both,
        DE,
        "Impressum\n\n{org#1}\n{address_ml#1}\n\n{role_heading}: {person#2}, {person#3}\nTelefon: {phone#4}\nE-Mail: {email#4}\n\nRegistergericht: {org_registry#5}\nRegisternummer: {neg_register_no}\n\n{sentence_local} {sentence_local}",
    ),
    (
        Letterhead,
        372,
        NoAddr,
        DE,
        "Das {org_gov#1} lädt am {date} zu einer Informationsveranstaltung ein. {person#2}, {title_local} der {org#3}, stellt die Ergebnisse vor. {sentence_local}",
    ),
    (
        Letterhead,
        373,
        NoAddr,
        DE,
        "Pressekontakt\n\n{org_gov#1}\nPressestelle\n{person#2}\nTelefon: {phone#2}\nE-Mail: {email#2}\n\n{sentence_local}",
    ),
    (
        Letterhead,
        374,
        NoAddr,
        DE,
        "{sentence_local} Die {org#1} wird vertreten durch {person#2} und {person#3}. {sentence_local} Seit {neg_date} ist {person_last#2} im Amt.",
    ),
    (
        Letterhead,
        375,
        NoPers,
        DE,
        "Die Veranstaltung findet in der {address#1} statt. {sentence_local} Veranstalter ist die {org#2} in Zusammenarbeit mit dem {org_gov#3}.",
    ),
    (
        Letterhead,
        376,
        NoAddr,
        DE,
        "Foto: {org_acr#1} / {person#2}\n\n{sentence_local} {sentence_local}\n\nIm Auftrag\n{person_last#3}",
    ),
    (
        Letterhead,
        377,
        NoAddr,
        DE,
        "Agenturen: {org_list}\n\n{sentence_local}\n\nVerantwortlich: {person#1}",
    ),
    (
        Letterhead,
        378,
        Both,
        DE,
        "{org#1}\n{address_ml#2}\n\nRechnung {neg_order} vom {neg_date}\n{product} {neg_price}\nGesamtbetrag {neg_price}\n\nBei Fragen: {person#3}, {phone#3}",
    ),
    (
        Letterhead,
        379,
        NoAddr,
        DE,
        "Impressum\n\n{org#1}\nVertreten durch: {person#2}\nRegistergericht: {org_registry#3}\n\n{role_heading}\n{person_list}",
    ),
    (
        Letterhead,
        380,
        NoPers,
        GB,
        "{org_gov#1}\n\nAddress\n\n{address_ml#2}\n\nTelephone\n\n{phone#3}\n({org_unit#4} - {neg_hours})\n\nEmail\n\n{email#5}\n\nOpening times\n\n{neg_hours}",
    ),
    (
        Letterhead,
        381,
        NoAddr,
        GB,
        "{role_heading}\n\n{person#1}\n\n{title}\n\n{person#2}\n\n{title}\n\n{person#3}\n\n{title}",
    ),
    (
        Letterhead,
        382,
        NoPers,
        GB,
        "Write to us:\n\n{org_gov#1}\n{address_ml#2}\n\nOnly contact {org_acr#1} if you cannot find the answer online.",
    ),
    (
        Letterhead,
        383,
        NoPers,
        GB,
        "{org#1}\nRegistered charity number {neg_digits}\n\nContact us\n\n{address_ml#2}\n\nSupporter Care: {phone#3}",
    ),
    (
        Letterhead,
        389,
        NoPers,
        GB,
        "{org_gov#1}\n\nWrite to\n\n{address_ml#5}\n\nTelephone\n\n{phone#2}\n({org_unit#3} - {neg_hours})\n\nEmail\n\n{email#4}",
    ),
    (
        Prose,
        70,
        NoPers,
        US,
        "FOR FURTHER INFORMATION CONTACT: {org_unit#1}, {org_gov#2}, {address#3}; telephone {phone#4}. The {org_acr:poss#2} reading room hours are {neg_hours}.",
    ),
    (
        Prose,
        71,
        NoAddr,
        US,
        "{org_acr#1} initiated the review on {date}. {person#2}, {title}, {org_unit#3}, can be reached at {phone#2}. {org_acr#1} Control Number {neg_digits}.",
    ),
];
