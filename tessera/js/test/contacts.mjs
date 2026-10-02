// Run with node or bun after `just wasm`; offsets must slice the original JS string.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createTessera } from "../../pkg/index.js";

const root = new URL("../../../", import.meta.url);
const modelBytes = await readFile(new URL("models/tessera-v1.safetensors", root));
const integrity = (await readFile(new URL("models/tessera-v1.sha256", root), "utf8")).trim();

const DOC =
  "Thanks, see you on Monday.\n\nJordan Avery, Project Coordinator\nAcme Corporation\n500 Main St, Springfield, IL 62701\n(202) 555-0199\njordan@acme.example";
const WIDE_DOC = `😀 Café\u0301: ${DOC}`;
const spans = (e) => [e.kind, e.text, e.start, e.end];

function expectedSpan(input, kind, text) {
  const start = input.indexOf(text);
  assert.ok(start >= 0, `missing expected ${kind}: ${text}`);
  return [kind, text, start, start + text.length];
}

function members(contact) {
  return [contact.person, contact.org, ...contact.addresses, ...contact.emails, ...contact.phones].filter(Boolean);
}

function assertSlices(input, entity) {
  assert.equal(input.slice(entity.start, entity.end), entity.text);
  for (const k of entity.components ?? []) {
    assert.equal(input.slice(k.start, k.end), k.text, k.label);
  }
}

function assertContact(input, out) {
  assert.deepEqual(out.unassigned, []);
  assert.equal(out.contacts.length, 1);
  const c = out.contacts[0];
  assert.deepEqual(spans(c.person), expectedSpan(input, "person", "Jordan Avery"));
  assert.deepEqual(spans(c.org), expectedSpan(input, "org", "Acme Corporation"));
  assert.deepEqual(c.addresses.map(spans), [expectedSpan(input, "address", "500 Main St, Springfield, IL 62701")]);
  assert.deepEqual(c.phones.map(spans), [expectedSpan(input, "phone", "(202) 555-0199")]);
  assert.deepEqual(c.emails.map(spans), [expectedSpan(input, "email", "jordan@acme.example")]);
  assert.equal(c.start, c.person.start);
  assert.equal(c.end, c.emails[0].end);
  assert.equal(c.phones[0].normalized, "+12025550199");
  assert.equal(c.phones[0].region, "US");
  assert.equal(c.emails[0].normalized, "jordan@acme.example");
  assert.deepEqual(
    c.addresses[0].components.map((k) => [k.label, k.text, k.start, k.end]),
    [
      expectedSpan(input, "house_number", "500"),
      expectedSpan(input, "road", "Main St"),
      expectedSpan(input, "city", "Springfield"),
      expectedSpan(input, "region", "IL"),
      expectedSpan(input, "postcode", "62701"),
    ],
  );
  const entities = members(c);
  assert.equal(c.confidence, Math.min(...entities.map((e) => e.confidence)), "signature confidence uses its weakest member");
  assert.ok(c.confidence >= 0.5 && c.confidence <= 1, `contact confidence ${c.confidence}`);
  assert.equal(c.reviewRecommended, c.confidence < 0.85);
  for (const e of entities) {
    assert.equal(e.reviewRecommended, e.confidence < 0.85, `${e.kind}: confidence band`);
    assertSlices(input, e);
  }
  return c;
}

const tessera = await createTessera({ modelBytes, integrity });
const plainContact = assertContact(DOC, await tessera.extractContacts(DOC, { countryHint: ["US"] }));
const wideContact = assertContact(WIDE_DOC, await tessera.extractContacts(WIDE_DOC, { countryHint: ["US"] }));
const shift = WIDE_DOC.indexOf(DOC);
assert.notEqual(shift, Buffer.byteLength(WIDE_DOC.slice(0, shift)), "prefix must distinguish UTF-16 from UTF-8");
for (const [i, entity] of members(wideContact).entries()) {
  const plain = members(plainContact)[i];
  assert.equal(entity.start, plain.start + shift);
  assert.equal(entity.end, plain.end + shift);
}

const rules = await createTessera({ kinds: ["email", "phone"] });
assert.deepEqual(rules.kinds, ["email", "phone"]);
for (const input of [DOC, WIDE_DOC]) {
  const out = await rules.extractContacts(input, { countryHint: ["US"] });
  assert.deepEqual(out.contacts, []);
  assert.deepEqual(out.unassigned.map(spans), [
    expectedSpan(input, "phone", "(202) 555-0199"),
    expectedSpan(input, "email", "jordan@acme.example"),
  ]);
  for (const e of out.unassigned) assertSlices(input, e);
}

tessera.dispose();
rules.dispose();
await assert.rejects(tessera.extractContacts(DOC), { code: "DISPOSED" });
await assert.rejects(rules.extractContacts(DOC), { code: "DISPOSED" });
await assert.rejects(createTessera({ modelBytes, integrity: "sha256-0000" }), { code: "CHECKSUM_MISMATCH" });
console.log("contacts: US grouping and Unicode offsets ok");
