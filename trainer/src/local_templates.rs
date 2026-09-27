//! Templates written for particular countries: sentences in the country's language around
//! the entities, the page layouts of its public bodies, and the name shapes its documents use.
//! US public notices and English news repeat surnames and acronyms without their definition.

use crate::generate::{Category, Family};
use Category::{NoAddressWithPerson as NoAddr, NoPersonWithAddress as NoPers};
use Family::*;

type Local = (Family, u32, Category, &'static [&'static str], &'static str);

const US: &[&str] = &["US"];
const NEWS: &[&str] = &["US"];

/// The country templates as `(family, id, category, countries, text)`; `templates::all` turns
/// them into `Template`s beside the shared ones.
pub const TEMPLATES: &[Local] = &[
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
        55,
        NoAddr,
        NEWS,
        "{person#1} met {person#2} on {date}. {person_last#1} told reporters that the talks with {person_last#2} were constructive, while {person_last:poss#2} office declined to comment. {sentence} {person_last#1} later left for {neg_place}.",
    ),
    (
        Prose,
        56,
        NoAddr,
        NEWS,
        "{org_party#1} chair {person#2} called on {person#3} to resign. \u{201c}{sentence}\u{201d} {person_last#2} said. {person_last#3} rejected the call, and {person_last:poss#3} spokesperson said {person_last#2} had misread the report.",
    ),
    (
        Prose,
        57,
        NoAddr,
        NEWS,
        "Read more:\n\n{date} \u{2013} {person_last#1} Meets {person_last#2} in {neg_place}\n{date} \u{2013} {person#3} Named {title}\n\n{sentence} {person#1} and {person#2} discussed trade. {person_last#1} thanked {person_last#2}.",
    ),
    (
        Prose,
        58,
        NoAddr,
        NEWS,
        "{person#1} told {org_media#2} on {date} that {org_party#3} would not back the bill. In an interview with {org_media#4}, {person_last#1} said {org_party#3} lawmakers had met {org_acr#5} officials. {sentence}",
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
