//! Detector document templates and the word lists their slots draw from.
//!
//! A template is plain text with `{slot}` or `{slot#n}` placeholders; `n` links slots within
//! one document (an email derived from a person, an org named twice). Ids are
//! `family_index * 100 + n`. The negative lists hold words that look like entities but are
//! not labelled: places and months that are also first names, companies named after people.

use anyhow::bail;

use crate::generate::{Category, Family, Slot};
use Category::{Both, NoAddressWithPerson as NoAddr, NoPersonWithAddress as NoPers, Nothing};
use Family::*;

/// One document template.
#[derive(Debug, Clone, Copy)]
pub struct Template {
    pub family: Family,
    pub id: u32,
    pub category: Category,
    pub text: &'static str,
}

const TEMPLATES: &[(Family, u32, Category, &str)] = &[
    (
        Prose,
        0,
        Both,
        "{sentence} I spoke with {person#1} from {org#1} yesterday; their office is at {address#1} and you can reach them on {phone#1}. {sentence}",
    ),
    (
        Prose,
        1,
        Both,
        "{person} moved the meeting to {date}. Send the paperwork to {address_ml} before then. {sentence}",
    ),
    (
        Prose,
        2,
        NoAddr,
        "{sentence} {person_eponymous} joined {org_eponymous} in {neg_date}; the board also welcomed {person}. {sentence}",
    ),
    (
        Prose,
        3,
        NoAddr,
        "In {neg_place}, {person} runs a workshop with {org}. {neg_word_name} is their busiest month, and {neg_word_name} said sales doubled.",
    ),
    (
        Prose,
        4,
        NoPers,
        "{org} opened a second site at {address}. {neg_road_sentence} {sentence}",
    ),
    (
        Prose,
        5,
        NoPers,
        "Parcels for {org_school} go to {address_person_street}. {neg_partial_location} {sentence}",
    ),
    (
        Prose,
        6,
        Nothing,
        "{sentence} The invoice total came to {neg_price} on {neg_date}, reference {neg_order}. {neg_road_sentence}",
    ),
    (
        Prose,
        7,
        Nothing,
        "{sentence} Our server at {neg_ip} rejected the upload; see {neg_url} and ping {neg_handle}. {sentence}",
    ),
    (
        Prose,
        8,
        Both,
        "Dear {person#1}, thank you for your order {neg_order}. It ships to {address_ml#1} and we will text {phone#1} on dispatch. {sentence}",
    ),
    (
        Prose,
        9,
        Both,
        "{sentence} {person} of {org} lives at {address}; write to {email} or call {phone}. {neg_word_name} and {neg_place} are not people. {sentence}",
    ),
    (
        EmailBody,
        100,
        Both,
        "{greeting} {person_first#1},\n\n{sentence} {sentence}\n\nCould you confirm the delivery address is still {address#1}?\n\n{closing},\n{person#2}\n{org#2}\n{phone#2}",
    ),
    (
        EmailBody,
        101,
        Both,
        "{greeting},\n\n{sentence}\n\nContact at the client is {person#1} ({email#1}, {phone#1}). Their registered office: {address_ml#1}.\n\n{closing},\n{person_first#2}",
    ),
    (
        EmailBody,
        102,
        NoAddr,
        "{greeting} {person_first#1},\n\n{sentence} {person#2} from {org#2} will join on {date}. {sentence}\n\n{closing},\n{person#1}",
    ),
    (
        EmailBody,
        103,
        NoAddr,
        "Hi all,\n\n{sentence} The {product} ticket is with {person}; the tracking number is {neg_digits}. {sentence}\n\n{closing},\n{person}",
    ),
    (
        EmailBody,
        104,
        NoPers,
        "{greeting},\n\nPlease redirect the shipment to {address_ml}. The old address is no longer valid. {sentence}\n\nThanks,\n{org}",
    ),
    (
        EmailBody,
        105,
        NoPers,
        "{greeting},\n\n{org} moved to {address} on {date}. {neg_road_sentence}\n\n{closing},\nThe team at {org}",
    ),
    (
        EmailBody,
        106,
        Nothing,
        "{greeting},\n\n{sentence} {sentence} The total is {neg_price}; payment reference {neg_order}. {sentence}\n\n{closing}",
    ),
    (
        EmailBody,
        107,
        Nothing,
        "Hi,\n\n{sentence} Please stop retrying from {neg_ip}; the endpoint {neg_url} is rate limited. {sentence}\n\nThanks",
    ),
    (
        EmailBody,
        108,
        Both,
        "{greeting} {person_first#1},\n\nAs discussed, the goods go to {address#1}. Invoice to {org#1} at {email#1}. {sentence}\n\n{closing},\n{person#2}\n{title}",
    ),
    (
        EmailBody,
        109,
        Both,
        "{greeting},\n\n{sentence}\n\nOn {date}, {person} wrote that {org} had relocated to {address}. {neg_word_name} disagreed. {sentence}\n\n{closing}",
    ),
    (
        Signature,
        200,
        Both,
        "{sentence}\n\n{closing},\n\n{person#1}\n{title}\n{org#1}\n{address_ml#1}\n{phone#1}\n{email#1}",
    ),
    (
        Signature,
        201,
        Both,
        "{closing},\n{person#1} | {title} | {org#1}\n{address#1} | T {phone#1} | {email#1}",
    ),
    (
        Signature,
        202,
        NoAddr,
        "{sentence}\n\n{closing}\n{person#1}\n{org#1}\nM {phone#1}\n{email#1}",
    ),
    (
        Signature,
        203,
        NoAddr,
        "--\n{person#1}\n{title}, {org}\n{email#1} · {neg_url}",
    ),
    (
        Signature,
        204,
        NoPers,
        "{sentence}\n\n{closing},\n{org#1}\n{address_ml#1}\n{phone#1}\n{email#1}",
    ),
    (
        Signature,
        205,
        NoPers,
        "{org#1}\n{address#1}\nTel {phone_local#1} · Fax {phone_local#1}\n{neg_url}",
    ),
    (
        Signature,
        206,
        Nothing,
        "{sentence}\n\n{closing},\nCustomer Care\nRef {neg_order}\n{neg_url}",
    ),
    (
        Signature,
        207,
        Nothing,
        "-- \nSent from my phone\nCase {neg_digits}\n{neg_url}",
    ),
    (
        Signature,
        208,
        Both,
        "{closing},\n{person#1}\n{org_eponymous#1}\n{address_person_street#1}\n{phone#1}",
    ),
    (
        Signature,
        209,
        Both,
        "{closing}\n{person#1}\n{title}\n{org_school#1}\n{address_ml#1}\n{phone#1} | {email#1} | {neg_handle}",
    ),
    (
        Letterhead,
        300,
        Both,
        "{org#1}\n{address_ml#1}\n{phone#1} · {email#1}\n\n{date}\n\n{person#2}\n{address_ml#2}\n\nDear {person_first#2},\n\n{sentence} {sentence}\n\n{closing},\n{person#3}\n{title}",
    ),
    (
        Letterhead,
        301,
        Both,
        "{org#1} | {address#1} | {phone#1}\n\n{date}\n\nAttn: {person#2}\n{org#2}\n{address#2}\n\n{sentence}\n\n{closing},\n{person#3}",
    ),
    (
        Letterhead,
        302,
        NoAddr,
        "{org#1}\n{phone#1} · {email#1}\n\n{date}\n\nTo whom it may concern,\n\n{sentence} {person#2} has been employed by {org#1} since {neg_date}.\n\n{closing},\n{person#3}\n{title}",
    ),
    (
        Letterhead,
        303,
        NoAddr,
        "{org}\n{neg_url}\n\n{date}\n\nDear {person},\n\n{sentence} Your reference is {neg_order}.\n\n{closing},\n{person}",
    ),
    (
        Letterhead,
        304,
        NoPers,
        "{org#1}\n{address_ml#1}\n{phone#1}\n\n{date}\n\nDear Customer,\n\n{sentence} Return the item to {address#2}.\n\n{closing},\n{org#1}",
    ),
    (
        Letterhead,
        305,
        NoPers,
        "{org#1}\nRegistered office: {address#1}\nVAT {neg_digits}\n\n{sentence}",
    ),
    (
        Letterhead,
        306,
        Nothing,
        "INVOICE\n\nNumber {neg_order}\nDate {neg_date}\nDue {neg_date}\n\n{sentence}\nTotal {neg_price}",
    ),
    (
        Letterhead,
        307,
        Nothing,
        "NOTICE\n\n{sentence} {sentence}\n\nReference {neg_order}\nIssued {neg_date}",
    ),
    (
        Letterhead,
        308,
        Both,
        "{org_eponymous#1}\n{address_person_street#1}\n{phone#1}\n\n{date}\n\n{person#2}\n{address_ml#2}\n\n{sentence}\n\n{closing},\n{person#3}",
    ),
    (
        Letterhead,
        309,
        Both,
        "{org#1} · {address#1}\n\n{date}\n\nDear {person#2},\n\n{sentence} Please confirm at {email#1} or {phone#1}.\n\n{closing},\n{person#3}, {title}",
    ),
    (
        Invoice,
        400,
        Both,
        "Invoice {neg_order}\nDate: {neg_date}\n\nBill to:\n{person#1}\n{org#1}\n{address_ml#1}\n\nShip to:\n{address_ml#2}\n\nItem  {product}  {neg_price}\nTotal {neg_price}\n\nQuestions: {phone#1}",
    ),
    (
        Invoice,
        401,
        Both,
        "From: {org#1}, {address#1}\nTo: {person#2}, {address#2}\nInvoice {neg_order} of {neg_date}\n\n{product} .......... {neg_price}\nTotal .......... {neg_price}\nIBAN {neg_iban}",
    ),
    (
        Invoice,
        402,
        NoAddr,
        "Receipt {neg_order}\nCustomer: {person}\nEmail: {email}\n\n{product} {neg_price}\n{product} {neg_price}\nTotal {neg_price}",
    ),
    (
        Invoice,
        403,
        NoAddr,
        "Purchase order {neg_order}\nSupplier: {org}\nContact: {person}, {phone}\nDelivery date: {neg_date}\nAmount: {neg_price}",
    ),
    (
        Invoice,
        404,
        NoPers,
        "Invoice {neg_order}\nSupplier: {org#1}\n{address_ml#1}\nVAT {neg_digits}\n\nTotal {neg_price}\nPay to IBAN {neg_iban}",
    ),
    (
        Invoice,
        405,
        NoPers,
        "Delivery note\nDeliver to: {address_ml}\nOrder {neg_order}\nItems: 3 × {product}\nDate {neg_date}",
    ),
    (
        Invoice,
        406,
        Nothing,
        "Invoice {neg_order}\nDate {neg_date}\n{product} {neg_price}\n{product} {neg_price}\nSubtotal {neg_price}\nTax {neg_price}\nTotal {neg_price}",
    ),
    (
        Invoice,
        407,
        Nothing,
        "Statement {neg_order}\nPeriod {neg_date} to {neg_date}\nOpening balance {neg_price}\nClosing balance {neg_price}",
    ),
    (
        Invoice,
        408,
        Both,
        "Quote {neg_order}\nPrepared for {person#1} at {org#1}\n{address#1}\n{email#1}\n\n{product} {neg_price}\nValid until {neg_date}",
    ),
    (
        Invoice,
        409,
        Both,
        "Tax invoice\n{org#1}\n{address_ml#1}\nTo: {person#2}\n{address_ml#2}\nAmount due {neg_price} by {neg_date}\nContact {phone#1}",
    ),
    (
        Markdown,
        500,
        Both,
        "# Contact\n\n**{person#1}**, {title} at *{org#1}*\n\n- Address: {address#1}\n- Phone: {phone#1}\n- Email: [{email#1}](mailto:{email#1})\n\n{sentence}",
    ),
    (
        Markdown,
        501,
        Both,
        "## Suppliers\n\n| Supplier | Contact | Address |\n| --- | --- | --- |\n| {org#1} | {person#1} | {address#1} |\n| {org#2} | {person#2} | {address#2} |",
    ),
    (
        Markdown,
        502,
        NoAddr,
        "# Team\n\n1. {person} — {title}\n2. {person} — {title}\n3. {person} — {title}\n\nAll at {org}. {sentence}",
    ),
    (
        Markdown,
        503,
        NoAddr,
        "> {person} wrote:\n> {sentence}\n\nReplying from {org}. See {neg_url} and `{neg_handle}`.",
    ),
    (
        Markdown,
        504,
        NoPers,
        "## Where to send it\n\n{org} accepts returns at:\n\n```\n{address_ml}\n```\n\n{sentence}",
    ),
    (
        Markdown,
        505,
        NoPers,
        "### Offices\n\n- **{org#1}**: {address#1}\n- **{org#2}**: {address#2}",
    ),
    (
        Markdown,
        506,
        Nothing,
        "# Changelog\n\n- {neg_date}: {sentence}\n- {neg_date}: fixed {neg_url}\n- {neg_date}: bumped {product}",
    ),
    (
        Markdown,
        507,
        Nothing,
        "## Payment\n\nTotal **{neg_price}** due {neg_date}.\nReference `{neg_order}`\nIBAN `{neg_iban}`",
    ),
    (
        Markdown,
        508,
        Both,
        "**Ship to**\n\n{person#1}\n{address_ml#1}\n\n**Bill to**\n\n{org#2}\n{address_ml#2}\n\n_Contact_: {phone#1}",
    ),
    (
        Markdown,
        509,
        Both,
        "# {org#1}\n\n{sentence}\n\n## Contact\n\n{person#1} · {email#1} · {phone#1}\n{address#1}\n\n{neg_word_name} is a month, {neg_place} is a place.",
    ),
    (
        Table,
        600,
        Both,
        "Name\tCompany\tAddress\tPhone\tEmail\n{person#1}\t{org#1}\t{address#1}\t{phone#1}\t{email#1}\n{person#2}\t{org#2}\t{address#2}\t{phone#2}\t{email#2}",
    ),
    (
        Table,
        601,
        Both,
        "| {person#1} | {address#1} | {phone#1} |\n| {person#2} | {address#2} | {phone#2} |\n| {person#3} | {address#3} | {phone#3} |",
    ),
    (
        Table,
        602,
        NoAddr,
        "Contact\tRole\tPhone\n{person}\t{title}\t{phone}\n{person}\t{title}\t{phone}\n{person}\t{title}\t{phone}",
    ),
    (
        Table,
        603,
        NoAddr,
        "Vendor\tContact\tEmail\n{org}\t{person}\t{email}\n{org}\t{person}\t{email}",
    ),
    (
        Table,
        604,
        NoPers,
        "Site\tAddress\n{org}\t{address}\n{org}\t{address}\n{org}\t{address}",
    ),
    (
        Table,
        605,
        NoPers,
        "| Warehouse | {address} |\n| Returns | {address} |\n| Billing | {org} |",
    ),
    (
        Table,
        606,
        Nothing,
        "Item\tQty\tPrice\n{product}\t2\t{neg_price}\n{product}\t1\t{neg_price}\nTotal\t\t{neg_price}",
    ),
    (
        Table,
        607,
        Nothing,
        "Date\tReference\tAmount\n{neg_date}\t{neg_order}\t{neg_price}\n{neg_date}\t{neg_order}\t{neg_price}",
    ),
    (
        Table,
        608,
        Both,
        "Customer,Address,Phone\n{person#1},\"{address#1}\",{phone#1}\n{person#2},\"{address#2}\",{phone#2}",
    ),
    (
        Table,
        609,
        Both,
        "{person#1}\t{org_eponymous#1}\t{address_person_street#1}\t{phone#1}\n{person#2}\t{org_school#2}\t{address#2}\t{phone#2}",
    ),
    (
        Support,
        700,
        Both,
        "Customer: {sentence} My name is {person#1} and my order {neg_order} went to the wrong address.\nAgent: Sorry about that. Which address should it go to?\nCustomer: {address#1}. You can call me on {phone#1}.\nAgent: Done. {sentence}",
    ),
    (
        Support,
        701,
        Both,
        "[{date}] {person#1}: {sentence}\n[{date}] Support: Can you confirm the account email?\n[{date}] {person#1}: {email#1}, and the billing address is {address#1}.\n[{date}] Support: Thanks, updated.",
    ),
    (
        Support,
        702,
        NoAddr,
        "Customer: Hi, this is {person_first#1} from {org#1}. {sentence}\nAgent: Hello {person_first#1}. What is the best number to reach you?\nCustomer: {phone#1}.\nAgent: {sentence}",
    ),
    (
        Support,
        703,
        NoAddr,
        "Ticket {neg_order}\nReporter: {person}\nAssignee: {person}\nSummary: {sentence}\nComment: {neg_word_name} said the {product} is fine.",
    ),
    (
        Support,
        704,
        NoPers,
        "Customer: Please ship the replacement to {address_ml}.\nAgent: Confirmed. {sentence}\nCustomer: Thanks, {org} will sign for it.",
    ),
    (
        Support,
        705,
        NoPers,
        "Agent: Our {neg_place} branch is at {address}. {neg_road_sentence}\nCustomer: {sentence}",
    ),
    (
        Support,
        706,
        Nothing,
        "Customer: {sentence} The charge of {neg_price} on {neg_date} is wrong.\nAgent: I see reference {neg_order}. {sentence}\nCustomer: Thanks.",
    ),
    (
        Support,
        707,
        Nothing,
        "Agent: Is the device showing IP {neg_ip}?\nCustomer: Yes. {sentence}\nAgent: Try {neg_url}. {sentence}",
    ),
    (
        Support,
        708,
        Both,
        "Customer: I'm {person#1}, {title} at {org#1}. {sentence}\nAgent: Is the invoice address still {address#1}?\nCustomer: Yes, and send copies to {email#1}.",
    ),
    (
        Support,
        709,
        Both,
        "Agent: Thanks for calling {org#1}, this is {person_first#2}.\nCustomer: Hi, {person#1} here. My address is {address#1} and my number is {phone#1}.\nAgent: Got it. {sentence}",
    ),
    (
        Prose,
        10,
        Both,
        "{neg_heading}\n\n{sentence} Please contact {person#1} at {org#1}, {address#1}, or call {phone#1}.",
    ),
    (
        Prose,
        11,
        NoAddr,
        "{sentence} Payroll questions go to {neg_department}; {person#1} ({email#1}) handles everything else. {sentence}",
    ),
    (
        Prose,
        12,
        NoPers,
        "{neg_heading}\n\n{org#1} has moved its {neg_department} to {address#1}. {sentence}",
    ),
    (
        Prose,
        13,
        Nothing,
        "{neg_heading}\n\n{sentence} {neg_prompt}: see {neg_url} or quote {neg_order}. {sentence}",
    ),
    (
        Prose,
        14,
        Both,
        "{person#1} ({title}, {org#1}) confirmed that the {neg_department} at {address#1} stays open until {neg_date}. {sentence}",
    ),
    (
        Prose,
        15,
        NoAddr,
        "{sentence} {org#1} said {person#1} would handle it.\n\n{neg_prompt}\nWrite to {email#1}.",
    ),
    (
        Prose,
        19,
        Both,
        "{neg_heading}\n{sentence} Our contact there is {person#1} of {org#1}, {address_ml#1}; {phone#1}.",
    ),
    (
        EmailBody,
        110,
        Both,
        "From: {person#1} <{email#1}>\nSubject: {product} order {neg_order}\n\n{greeting},\n\n{sentence}\n\nPlease ship to:\n{person#2}\n{org#2}\n{address_ml#2}\n\n{closing},\n{person#1}",
    ),
    (
        EmailBody,
        111,
        NoAddr,
        "{greeting} {person_first#1},\n\n{sentence} I have copied {neg_department} and {person#2} ({email#2}).\n\n{closing},\n{person#3}\n{neg_department}\n{org#3}",
    ),
    (
        EmailBody,
        112,
        NoPers,
        "{greeting},\n\n{neg_heading}\n{sentence}\n\n{neg_prompt}\n{org#1}, {address#1}\n{phone#1}",
    ),
    (
        EmailBody,
        113,
        Nothing,
        "{greeting},\n\n{neg_header}\n{product}  2  {neg_price}\n{product}  1  {neg_price}\n\n{sentence}\n\n{closing},\n{neg_department}",
    ),
    (
        EmailBody,
        114,
        Both,
        "{greeting} {person_first#1},\n\n{neg_heading}\n{sentence} The site contact is {person#2}, {address#2}.\n\n{neg_heading}\n{sentence}\n\n{closing},\n{person#3}\n{title}, {org#3}",
    ),
    (
        EmailBody,
        115,
        NoAddr,
        "On {date}, {person#1} <{email#1}> wrote:\n> {sentence}\n> {closing},\n> {person#1}\n\n{greeting},\n{sentence}\n\n{closing},\n{person#2}",
    ),
    (
        EmailBody,
        119,
        Both,
        "From: {person#1} <{email#1}>\n\n{greeting},\n\n{neg_heading}\n{sentence} Deliver to {org#2}, {address#2}.\n\n{neg_prompt} {phone#1}\n\n{closing},\n{person#1}",
    ),
    (
        Signature,
        210,
        Both,
        "{closing},\n\n{person#1}\n{neg_department}\n{org#1}\n{address_ml#1}\n{neg_prompt} {phone#1}",
    ),
    (
        Signature,
        211,
        NoAddr,
        "{person#1} | {neg_department} | {org#1}\n{email#1} | {phone#1}",
    ),
    (
        Signature,
        212,
        NoPers,
        "{neg_department}\n{org#1}\n{address#1}\n{neg_prompt}: {phone#1} · {email#1}",
    ),
    (
        Signature,
        213,
        Nothing,
        "{closing},\n{neg_department}\n{neg_prompt} {neg_url}\nRef {neg_order}",
    ),
    (
        Signature,
        214,
        Both,
        "{closing}\n\n{person#1}\n{title} · {neg_department}\n\n{org#1}\n{address#1}\nT {phone#1}\nE {email#1}",
    ),
    (
        Signature,
        215,
        NoAddr,
        "--\n{person#1}\n{org#1}\n\n{neg_prompt}\n{phone#1}\n{email#1}",
    ),
    (
        Signature,
        219,
        Both,
        "{closing},\n{person#1}\n{neg_department}, {org#1}\n{address_ml#1}\n{neg_prompt} {phone#1}",
    ),
    (
        Letterhead,
        310,
        Both,
        "{org#1}\n{neg_department}\n{address_ml#1}\n\n{date}\n\nAttn: {person#2}, {neg_department}\n{org#2}\n{address_ml#2}\n\n{sentence}\n\n{closing},\n{person#3}",
    ),
    (
        Letterhead,
        311,
        NoPers,
        "{org#1}\n{address#1}\n\n{neg_heading}\n{sentence}\n\n{neg_prompt}\n{phone#1} · {email#1}",
    ),
    (
        Letterhead,
        312,
        NoAddr,
        "{org#1} · {neg_department}\n\n{date}\n\nDear {person#2},\n\n{sentence}\n\n{neg_prompt} {email#1}\n\n{closing},\n{person#3}",
    ),
    (
        Letterhead,
        313,
        Nothing,
        "{neg_department}\n\n{date}\n\n{neg_heading}\n{sentence} {sentence}\n\nReference {neg_order}",
    ),
    (
        Letterhead,
        314,
        Both,
        "{org#1}\n{address#1}\n\nTo: {org#2}\nAttn: {person#2}\n{address_ml#2}\n\n{neg_heading}\n{sentence}\n\n{closing},\n{person#3}, {neg_department}",
    ),
    (
        Letterhead,
        315,
        NoPers,
        "{org#1}\n{address_ml#1}\n\n{neg_department}\n{org#2}\n{address_ml#2}\n\n{sentence}",
    ),
    (
        Letterhead,
        319,
        Both,
        "{org#1}\n{address_ml#1}\n\n{person#2}\n{neg_department}\n{org#2}\n{address_ml#2}\n\n{sentence}\n\n{closing},\n{person#3}",
    ),
    (
        Invoice,
        410,
        Both,
        "Invoice {neg_order}\n\nBill to:\n{org#1}\nAttn: {person#1}, {neg_department}\n{address_ml#1}\n\n{neg_header}\n{product}  3  {neg_price}  {neg_price}\n{product}  1  {neg_price}  {neg_price}\n\nTotal due {neg_price}\n\n{neg_prompt} {person#2} on {phone#2}.",
    ),
    (
        Invoice,
        411,
        NoPers,
        "{org#1}\n{address_ml#1}\n\nInvoice {neg_order}   Date {neg_date}\n\n{neg_header}\n{product}\t1\t{neg_price}\n{product}\t4\t{neg_price}\n\nSubtotal {neg_price}\nTotal {neg_price}\n\n{neg_prompt}: {email#1}",
    ),
    (
        Invoice,
        412,
        NoAddr,
        "Credit note {neg_order}\nCustomer: {org#1}\nContact: {person#1} ({neg_department})\n\n{neg_header}\n{product}  1  {neg_price}\n\n{neg_prompt} {email#1}",
    ),
    (
        Invoice,
        413,
        Nothing,
        "Statement {neg_order}\n\n{neg_header}\n{neg_date}  {product}  {neg_price}\n{neg_date}  {product}  {neg_price}\n\nBalance {neg_price}\nPay to IBAN {neg_iban}",
    ),
    (
        Invoice,
        414,
        Both,
        "{org#1}\n{address#1}\n\nINVOICE {neg_order}\n\nShip to:\n{person#2}\n{address_ml#2}\n\n{neg_header}\n{product}  2  {neg_price}\n\n{neg_department}: {phone#1}",
    ),
    (
        Invoice,
        415,
        NoPers,
        "Remit to:\n{org#1}\n{neg_department}\n{address_ml#1}\n\n{neg_header}\n{product}  5  {neg_price}\nTotal {neg_price}",
    ),
    (
        Invoice,
        419,
        Both,
        "Invoice {neg_order}\n\nFrom:\n{org#1}\n{address_ml#1}\n\nTo:\n{person#2}\n{org#2}\n{address_ml#2}\n\n{neg_header}\n{product}  1  {neg_price}\n\n{neg_prompt} {phone#1}",
    ),
    (
        Table,
        610,
        NoAddr,
        "{neg_header}\n{person#1}\t{title}\t{phone#1}\t{email#1}\n{person#2}\t{title}\t{phone#2}\t{email#2}",
    ),
    (
        Table,
        611,
        Both,
        "{neg_heading}\n\nName\tCompany\tAddress\n{person#1}\t{org#1}\t{address#1}\n{person#2}\t{org#2}\t{address#2}",
    ),
    (
        Table,
        612,
        NoPers,
        "{neg_header}\n{org#1}\t{address#1}\t{phone#1}\n{org#2}\t{address#2}\t{phone#2}",
    ),
    (
        Table,
        613,
        Nothing,
        "{neg_heading}\n\n{neg_header}\n{product}\t{neg_digits}\t{neg_price}\n{product}\t{neg_digits}\t{neg_price}",
    ),
    (
        Table,
        614,
        NoAddr,
        "{neg_department}\n{neg_header}\n{person#1}\t{neg_date}\t{email#1}\n{person#2}\t{neg_date}\t{email#2}",
    ),
    (
        Table,
        615,
        Both,
        "| Name | Organisation | Address |\n| --- | --- | --- |\n| {person#1} | {org#1} | {address#1} |\n| {person#2} | {org#2} | {address#2} |",
    ),
    (
        Table,
        619,
        NoAddr,
        "{neg_heading}\n{neg_header}\n{person#1}\t{org#1}\t{phone#1}\n{person#2}\t{org#2}\t{email#2}",
    ),
    (
        Support,
        710,
        Both,
        "Ticket {neg_order}\n{neg_department}\n\nCustomer: {person#1}\nCompany: {org#1}\nAddress: {address#1}\nPhone: {phone#1}\n\n{sentence}",
    ),
    (
        Support,
        711,
        NoAddr,
        "Agent ({neg_department}): {sentence}\nCustomer: This is {person#1} from {org#1}. You can email me at {email#1}.\nAgent: {sentence}",
    ),
    (
        Support,
        712,
        NoPers,
        "{neg_prompt}\n\n{org#1} support\n{address#1}\n{phone#1}\n\n{sentence}",
    ),
    (
        Support,
        713,
        Nothing,
        "Case {neg_digits} assigned to {neg_department}.\n{neg_prompt} Reply to this message quoting {neg_order}.\n{sentence}",
    ),
    (
        Support,
        714,
        Both,
        "Customer: Hi, I'm {person#1}.\nAgent: Thanks {person_first#1}, could you confirm your address?\nCustomer: {address#1}, and my number is {phone#1}.\nAgent: {sentence}",
    ),
    (
        Support,
        715,
        NoAddr,
        "Escalated to {neg_department}\nOwner: {person#1} <{email#1}>\nRequester: {person#2} ({org#2})\n\n{sentence}",
    ),
    (
        Support,
        719,
        Both,
        "Ticket {neg_order} · {neg_department}\n\nRequester: {person#1}\nOrganisation: {org#1}\nShipping address: {address_ml#1}\nCallback: {phone#1}",
    ),
];

impl Template {
    /// Whether this template is only ever used for test documents: the two held-out
    /// families whole, and every id ending in 9 in the other six.
    pub fn test_only(&self) -> bool {
        self.family.held_out() || self.id % 10 == 9
    }

    /// The slots in order, each with its link group (0 when unlinked).
    pub fn slots(&self) -> anyhow::Result<Vec<(Slot, u8)>> {
        let mut out = Vec::new();
        let mut rest = self.text;
        while let Some(open) = rest.find('{') {
            let Some(len) = rest[open..].find('}') else {
                bail!("unclosed slot in template {}", self.id);
            };
            let inner = &rest[open + 1..open + len];
            let (name, group) = match inner.split_once('#') {
                Some((n, g)) => (n, g.parse::<u8>()?),
                None => (inner, 0),
            };
            let Some(slot) = Slot::parse(name) else {
                bail!("unknown slot {name} in template {}", self.id);
            };
            out.push((slot, group));
            rest = &rest[open + len + 1..];
        }
        Ok(out)
    }

    /// The category the slots imply: whether any slot yields a person and any an address.
    pub fn derived_category(&self) -> anyhow::Result<Category> {
        let kinds: Vec<_> = self
            .slots()?
            .into_iter()
            .filter_map(|(s, _)| s.kind())
            .collect();
        let person = kinds.contains(&tessera::Kind::Person);
        let address = kinds.contains(&tessera::Kind::Address);
        Ok(match (person, address) {
            (true, true) => Category::Both,
            (true, false) => Category::NoAddressWithPerson,
            (false, true) => Category::NoPersonWithAddress,
            (false, false) => Category::Nothing,
        })
    }
}

/// Every template, after checking that each parses and that its declared category is the
/// one its slots imply.
pub fn all() -> anyhow::Result<Vec<Template>> {
    TEMPLATES
        .iter()
        .map(|&(family, id, category, text)| {
            let t = Template {
                family,
                id,
                category,
                text,
            };
            let derived = t.derived_category()?;
            if derived != category {
                bail!("template {id} is declared {category:?} but its slots make it {derived:?}");
            }
            Ok(t)
        })
        .collect()
}

pub const NEG_PLACES: &[&str] = &[
    "Georgia",
    "Jordan",
    "Washington",
    "Victoria",
    "Chad",
    "Paris",
    "Florence",
    "Sydney",
    "Berlin",
    "Tbilisi",
    "Kyoto",
    "Adelaide",
    "Lincoln",
    "Madison",
    "Jackson",
    "Austin",
    "Charlotte",
    "Orlando",
    "Hamburg",
    "Batumi",
];

pub const NEG_WORD_NAMES: &[&str] = &[
    "April",
    "May",
    "June",
    "August",
    "Will",
    "Bill",
    "Mark",
    "Grace",
    "Hope",
    "Chase",
    "Pat",
    "Sue",
    "Rob",
    "Art",
    "Ray",
    "Sunny",
    "Wednesday",
    "Summer",
    "Autumn",
    "Monday",
];

/// Labelled org when used as `org_eponymous`.
pub const EPONYMOUS_COMPANIES: &[&str] = &[
    "Ford", "Siemens", "Dior", "Boeing", "Dell", "Heinz", "Tesla", "Disney", "Ferrari", "Kellogg",
    "Philips", "Bosch", "Cadbury", "Mars", "Honda", "Toyota", "Suzuki", "Porsche", "Hilton",
    "Marriott",
];

/// Labelled person when used as `person_eponymous`, always after a given name.
pub const EPONYMOUS_SURNAMES: &[&str] = &[
    "Ford", "Siemens", "Dior", "Dell", "Heinz", "Disney", "Kellogg", "Bosch", "Honda", "Hilton",
];

/// Labelled org when used as `org_school`.
pub const PERSON_NAMED_INSTITUTIONS: &[&str] = &[
    "Abraham Lincoln High School",
    "Johns Hopkins University",
    "Stanford University",
    "Humboldt-Universität zu Berlin",
    "Ivane Javakhishvili Tbilisi State University",
    "Keio University",
    "Max Planck Institute",
    "Wellcome Trust",
    "Rockefeller Foundation",
    "Carnegie Hall",
];

/// Streets named after people, per country, and the city used when a sampled address has
/// none.
pub const PERSON_NAMED_STREETS: &[(&str, &str, &[&str])] = &[
    (
        "US",
        "Washington",
        &[
            "Washington Street",
            "Jefferson Road",
            "Martin Luther King Jr. Boulevard",
            "Lincoln Avenue",
        ],
    ),
    (
        "GB",
        "London",
        &[
            "Churchill Way",
            "Queen Elizabeth Street",
            "Nelson Road",
            "Wellington Place",
        ],
    ),
    (
        "DE",
        "Berlin",
        &[
            "Karl-Marx-Allee",
            "Friedrich-Ebert-Straße",
            "Goethestraße",
            "Schillerstraße",
        ],
    ),
    (
        "GE",
        "Tbilisi",
        &[
            "Rustaveli Avenue",
            "Chavchavadze Avenue",
            "რუსთაველის გამზირი",
            "ჭავჭავაძის გამზირი",
        ],
    ),
    ("JP", "Tokyo", &["Meiji-dori", "明治通り"]),
];

pub const NEG_HANDLES: &[&str] = &[
    "@nino_b",
    "@kavkaz_freight",
    "@support",
    "@jsmith88",
    "#ticket-4021",
];

pub const NEG_URLS: &[&str] = &[
    "https://status.example",
    "https://docs.example/api",
    "www.example.org/returns",
    "example.com/track",
];

pub const NEG_ROAD_SENTENCES: &[&str] = &[
    "We walked down the road to the station.",
    "Turn left at the next street.",
    "The avenue was closed for repairs.",
    "Head north on the highway for ten minutes.",
    "The lane behind the office floods in winter.",
];

pub const NEG_PARTIAL_LOCATIONS: &[&str] = &[
    "They are based in Tbilisi.",
    "The office is somewhere in Berlin.",
    "Shipping within Georgia takes two days.",
    "Most customers are in London.",
    "We serve all of Kansai.",
];

/// Phone-shaped numbers that are not phone numbers.
pub const NEG_DIGITS: &[&str] = &[
    "Tracking 1Z 12E 4F5 03 9876 2210",
    "Order 5530-1182-07",
    "Ref 20260922-0417",
    "Parcel 3S 7710 2291 44",
    "Ticket 55120-884",
];

pub const NEG_IBANS: &[&str] = &[
    "GB82 WEST 1234 5698 7654 32",
    "DE44 5001 0517 5407 3249 31",
    "GE60 NB00 0000 0123 4567 89",
];

pub const NEG_PRICES: &[&str] = &[
    "€1,240.00",
    "$89.99",
    "£12.50",
    "¥3,200",
    "4.500,00 €",
    "GEL 120",
    "USD 2,410.75",
];

pub const NEG_ORDERS: &[&str] = &[
    "INV-2025-1177",
    "#48213",
    "PO 77120-B",
    "SO-0091",
    "A-55213",
    "2026/0917",
];

/// Dates that open with a month named like a person, and plain numeric forms.
pub const NEG_DATES: &[&str] = &[
    "April 2021",
    "12 May 2024",
    "June 3, 2025",
    "August 2019",
    "2024-08-01",
    "03.11.2023",
    "March 2022",
];

pub const DATES: &[&str] = &[
    "22 September 2026",
    "2026-09-22",
    "09/22/2026",
    "3 October 2026",
    "14.10.2026",
    "November 5, 2026",
];

pub const PRODUCTS: &[&str] = &[
    "Model X-200",
    "Series 4 bracket",
    "cable set B",
    "Type 7 adapter",
    "wall mount kit",
];

/// Latin local parts for emails whose person has no Latin-script name.
pub const LOCAL_PARTS: &[&str] = &[
    "office", "info", "contact", "mail", "hello", "team", "desk", "admin",
];

/// Words for email domains that are not derived from an org.
pub const DOMAIN_WORDS: &[&str] = &[
    "northwind",
    "bluebird",
    "kavkaz",
    "lindenhof",
    "sakura",
    "harbor",
    "meridian",
    "alpine",
    "orchard",
    "summit",
];

/// Reserved email domain endings: reserved top-level and second-level domains only.
pub const EMAIL_SUFFIXES: &[&str] = &[".example", ".test", ".example.com", ".example.org"];

/// Table and invoice column headers: a line of capitalised words that is never a name.
pub const NEG_HEADERS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "Description    Quantity    Rate    Amount",
            "Product\tSKU\tQty\tPrice",
            "Service          Hours   Rate    Line total",
            "No.  Article  Units  Unit price  Sum",
            "Code    Item description    Qty    Net",
            "Date       Description        Debit    Credit",
            "Name\tRole\tPhone\tEmail",
            "Line  Part number  Ordered  Shipped",
        ],
    ),
    (
        "DE",
        &[
            "Pos.  Bezeichnung  Menge  Einzelpreis  Gesamt",
            "Artikel\tAnzahl\tPreis\tBetrag",
            "Leistung      Stunden   Satz   Summe",
        ],
    ),
    (
        "GE",
        &[
            "დასახელება   რაოდენობა   ფასი   ჯამი",
            "N\tპროდუქტი\tერთეული\tთანხა",
        ],
    ),
    (
        "JP",
        &[
            "品名　数量　単価　金額",
            "項目\t数量\t金額",
            "番号　品目　個数　小計",
        ],
    ),
];

/// Department and team names, written where a person or an organization often stands.
pub const NEG_DEPARTMENTS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "Customer Service",
            "Human Resources",
            "Procurement",
            "Facilities Management",
            "Billing Department",
            "Legal Department",
            "Payroll Team",
            "IT Service Desk",
            "Credit Control",
            "Logistics Team",
        ],
    ),
    (
        "DE",
        &[
            "Kundenservice",
            "Buchhaltung",
            "Personalabteilung",
            "Einkauf",
            "Rechtsabteilung",
        ],
    ),
    (
        "GE",
        &[
            "ბუღალტერია",
            "კადრების განყოფილება",
            "მომხმარებელთა სერვისი",
        ],
    ),
    (
        "JP",
        &["総務部", "経理部", "営業部", "人事部", "カスタマーサポート"],
    ),
];

/// Short prompts and labels standing alone before contact details.
pub const NEG_PROMPTS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "Need help?",
            "Any problems?",
            "Get in touch",
            "Still stuck?",
            "Contact us",
            "Reach us",
            "More information",
            "Support hours",
        ],
    ),
    (
        "DE",
        &[
            "Noch Fragen?",
            "Kontakt",
            "Hilfe benötigt?",
            "Erreichbarkeit",
        ],
    ),
    ("GE", &["კითხვები გაქვთ?", "დაგვიკავშირდით"]),
    ("JP", &["お問い合わせ", "ご不明な点は", "連絡先"]),
];

/// Section headings: regions, countries, and topics, often on a line of their own.
pub const NEG_HEADINGS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "France",
            "Italy",
            "Canada",
            "Australia",
            "Nordics",
            "Asia Pacific",
            "North America",
            "Head office",
            "Regional offices",
            "Next steps",
            "Overview",
            "Shipping schedule",
            "Project roles",
            "Key contacts",
        ],
    ),
    (
        "DE",
        &[
            "Österreich",
            "Schweiz",
            "Standorte",
            "Ansprechpartner",
            "Übersicht",
        ],
    ),
    ("GE", &["ბათუმი", "რეგიონები", "კონტაქტები"]),
    ("JP", &["関西", "九州", "拠点一覧", "担当者"]),
];

pub const HONORIFICS: &[&str] = &["Mr", "Ms", "Mrs", "Dr", "Prof."];

/// Greetings, closings, and job titles per country; English is mixed into every country.
pub const GREETINGS: &[(&str, &[&str])] = &[
    ("en", &["Hi", "Hello", "Dear all", "Good morning"]),
    (
        "DE",
        &["Guten Tag", "Hallo", "Sehr geehrte Damen und Herren"],
    ),
    ("GE", &["გამარჯობა", "სალამი"]),
    ("JP", &["お世話になっております", "こんにちは"]),
];

pub const CLOSINGS: &[(&str, &[&str])] = &[
    ("en", &["Regards", "Best", "Kind regards", "Many thanks"]),
    ("DE", &["Mit freundlichen Grüßen", "Viele Grüße"]),
    ("GE", &["პატივისცემით", "მადლობა"]),
    ("JP", &["よろしくお願いいたします", "敬具"]),
];

pub const TITLES: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "Sales Manager",
            "Head of Operations",
            "Office manager",
            "Senior accountant",
        ],
    ),
    ("DE", &["Geschäftsführer", "Vertriebsleiterin"]),
    ("GE", &["მთავარი ბუღალტერი", "დირექტორი"]),
    ("JP", &["営業部長", "総務課長"]),
];

/// Neutral sentences with no proper nouns and no digits; only the first word is capitalized.
pub const FILLER_SENTENCES: &[&str] = &[
    "Thanks again for the quick turnaround on this.",
    "Let me know if anything is unclear.",
    "We are happy to help with any questions.",
    "Please find the details below.",
    "The documents are attached for your review.",
    "Everything looks good on our side.",
    "We will follow up as soon as we hear back.",
    "Apologies for the delay in getting back to you.",
    "Hope you had a good weekend.",
    "The team has reviewed the request.",
    "We appreciate your patience while we sort this out.",
    "It was good to catch up last week.",
    "Please double check the figures before signing.",
    "The shipment left the warehouse this morning.",
    "We have updated our records accordingly.",
    "Feel free to forward this to your colleagues.",
    "The meeting notes are in the shared folder.",
    "We look forward to working with you.",
    "Nothing else is needed from you at this stage.",
    "The contract renews automatically each year.",
    "We have not received the signed copy yet.",
    "Let us know which option works best.",
    "The payment has been processed.",
    "Our office will be closed over the holidays.",
    "Please reply to this message to confirm.",
    "The replacement part is on its way.",
    "We noticed a small error in the last invoice.",
    "The new schedule starts next month.",
    "Thanks for flagging the issue so quickly.",
    "We are still waiting on the final numbers.",
    "The account has been set up as requested.",
    "Could you send the updated version when ready?",
    "This message was sent to the right person.",
    "The order was split into two parcels.",
    "We will call you if anything changes.",
    "Please keep this email for your records.",
    "The draft looks fine with one small change.",
    "Our support hours are listed on the website.",
    "The item arrived damaged and needs replacing.",
    "We confirmed the booking this afternoon.",
];

/// Georgian legal forms for organizations borrowed into a small GE pool: the Latin
/// abbreviations after the name, the Georgian ones before it.
pub const GE_LEGAL_FORMS: &[(&str, bool)] =
    &[("LLC", false), ("JSC", false), ("შპს", true), ("სს", true)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_parses_with_its_declared_category_and_a_unique_id() {
        let all = all().unwrap();
        assert_eq!(all.len(), 129);
        let mut ids: Vec<u32> = all.iter().map(|t| t.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 129);
        for t in &all {
            assert_eq!(t.id / 100, t.family as u32, "template {}", t.id);
        }
    }

    #[test]
    fn test_only_templates_are_held_out_families_and_ids_ending_in_nine() {
        let all = all().unwrap();
        assert_eq!(all.iter().filter(|t| t.test_only()).count(), 24);
        assert!(
            all.iter()
                .filter(|t| !t.test_only())
                .all(|t| { !t.family.held_out() && t.id % 10 != 9 })
        );
    }

    #[test]
    fn filler_sentences_have_no_digits_or_inner_capitals() {
        assert_eq!(FILLER_SENTENCES.len(), 40);
        for s in FILLER_SENTENCES {
            assert!(!s.chars().any(|c| c.is_ascii_digit()), "{s}");
            assert!(
                !s.split_whitespace()
                    .skip(1)
                    .any(|w| w.starts_with(char::is_uppercase)),
                "{s}"
            );
        }
    }
}
