// extractContacts through the built package: the spec's worked example, and its Georgian twin,
// whose byte and UTF-16 offsets differ for every entity. Run with node or bun after `just wasm`.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createTessera } from "../../pkg/index.js";

const root = new URL("../../../", import.meta.url);
const modelBytes = await readFile(new URL("models/tessera-v1.safetensors", root));
const integrity = (await readFile(new URL("models/tessera-v1.sha256", root), "utf8")).trim();

const DOC =
  "Thanks, see you on Monday.\n\nNino Beridze\nKavkaz Freight LLC\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456\nnino@kavkaz-freight.example";

const GEORGIAN =
  "მადლობა, ორშაბათს შევხვდებით.\n\nნინო ბერიძე\nშპს კავკაზ ფრეითი\nრუსთაველის გამზირი 14, თბილისი 0108, საქართველო\n+995 32 212 3456\nnino@kavkaz-freight.example";

const spans = (e) => [e.kind, e.text, e.start, e.end];

function members(contact) {
  return [contact.person, contact.org, ...contact.addresses, ...contact.emails, ...contact.phones].filter(Boolean);
}

function assertSlices(input, entity) {
  assert.equal(input.slice(entity.start, entity.end), entity.text);
  for (const k of entity.components ?? []) {
    assert.equal(input.slice(k.start, k.end), k.text, k.label);
  }
}

const tessera = await createTessera({ modelBytes, integrity });

const out = await tessera.extractContacts(DOC, { countryHint: ["GE"] });
assert.deepEqual(out.unassigned, []);
assert.equal(out.contacts.length, 1);
const c = out.contacts[0];
assert.equal(c.start, 28);
assert.equal(c.end, 147);
assert.deepEqual(spans(c.person), ["person", "Nino Beridze", 28, 40]);
assert.deepEqual(spans(c.org), ["org", "Kavkaz Freight LLC", 41, 59]);
assert.deepEqual(c.addresses.map(spans), [["address", "14 Rustaveli Avenue, Tbilisi 0108, Georgia", 60, 102]]);
assert.deepEqual(c.phones.map(spans), [["phone", "+995 32 212 3456", 103, 119]]);
assert.deepEqual(c.emails.map(spans), [["email", "nino@kavkaz-freight.example", 120, 147]]);
assert.equal(c.phones[0].normalized, "+995322123456");
assert.equal(c.phones[0].region, "GE");
assert.equal(c.emails[0].normalized, "nino@kavkaz-freight.example");
assert.deepEqual(
  c.addresses[0].components.map((k) => [k.label, k.text, k.start, k.end]),
  [
    ["house_number", "14", 60, 62],
    ["road", "Rustaveli Avenue", 63, 79],
    ["city", "Tbilisi", 81, 88],
    ["postcode", "0108", 89, 93],
    ["country", "Georgia", 95, 102],
  ],
);
assert.ok(c.confidence >= 0.85, `contact confidence ${c.confidence}`);
assert.equal(c.reviewRecommended, false);
for (const e of members(c)) assertSlices(DOC, e);

// Person and org are experimental in Georgian script, so only the rule-found entities and the
// parsed address are held to exact offsets; every offset returned must slice the string.
const ge = await tessera.extractContacts(GEORGIAN, { countryHint: ["GE"] });
assert.equal(ge.contacts.length, 1);
const g = ge.contacts[0];
assert.deepEqual(g.addresses.map(spans), [["address", "რუსთაველის გამზირი 14, თბილისი 0108, საქართველო", 61, 108]]);
assert.deepEqual(g.phones.map(spans), [["phone", "+995 32 212 3456", 109, 125]]);
assert.deepEqual(g.emails.map(spans), [["email", "nino@kavkaz-freight.example", 126, 153]]);
assert.equal(g.end, 153);
assert.equal(GEORGIAN.length, 153);
for (const e of [...members(g), ...ge.unassigned]) assertSlices(GEORGIAN, e);

const rules = await createTessera({ kinds: ["email", "phone"] });
const plain = await rules.extractContacts(DOC, { countryHint: ["GE"] });
assert.deepEqual(plain.contacts, []);
assert.deepEqual(plain.unassigned.map((e) => e.kind), ["phone", "email"]);

tessera.dispose();
rules.dispose();
console.log("contacts: ok");
