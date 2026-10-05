import assert from "node:assert/strict";
import test from "node:test";
import { fromUtf8Case, scoreCases, validateGold } from "./scoring.mjs";

const input = "Jane Doe\njane@example.org\nJohn Doe\njohn@example.org";
const entity = (kind, text) => ({ kind, text, start: input.indexOf(text), end: input.indexOf(text) + text.length });
const entities = [entity("person", "Jane Doe"), entity("email", "jane@example.org"), entity("person", "John Doe"), entity("email", "john@example.org")];
const gold = {
  name: "two contacts", input, entities,
  contacts: [{ person: 0, emails: [1] }, { person: 2, emails: [3] }], unassigned: [],
};
const extraction = { contacts: [{ person: entities[0], emails: [entities[1]] }, { person: entities[2], emails: [entities[3]] }], unassigned: [] };
const score = (detected = entities, output = extraction, c = gold) => scoreCases([{ gold: c, detected, extraction: output }]);

test("duplicate predictions and cards only match once", () => {
  const result = score([...entities, entities[0]], { ...extraction, contacts: [...extraction.contacts, extraction.contacts[0]] });
  assert.equal(result.detection.person.tp, 2);
  assert.equal(result.detection.person.fp, 1);
  assert.equal(result.detection.person.recall, 1);
  assert.equal(result.contactFields.person.fp, 1);
  assert.equal(result.exactCards.fp, 1);
});

test("wrong owner costs field precision and recall even when detection is perfect", () => {
  const result = score(entities, { contacts: [{ person: entities[0], emails: [entities[3]] }, { person: entities[2], emails: [entities[1]] }], unassigned: [] });
  assert.equal(result.detection.email.f1, 1);
  assert.equal(result.contactFields.email.tp, 0);
  assert.equal(result.contactFields.email.fp, 2);
  assert.equal(result.contactFields.email.fn, 2);
});

test("one-character boundary error is an exact false positive and a miss", () => {
  const broken = { ...entities[0], end: entities[0].end - 1, text: "Jane Do" };
  const result = score([broken, ...entities.slice(1)]);
  assert.equal(result.detection.person.fp, 1);
  assert.equal(result.detection.person.fn, 1);
  assert.equal(result.exactFalsePositiveDocumentRate.person, 1);
});

test("missing unassigned entities cost detection recall", () => {
  const c = { name: "unassigned", input, entities: [entities[1]], contacts: [], unassigned: [0] };
  const result = score([], { contacts: [], unassigned: [] }, c);
  assert.equal(result.detection.email.fn, 1);
  assert.equal(result.contactFields.email.gold, 0);
  assert.equal(result.allFieldsPass95, false);
  const detectedButDropped = score([entities[1]], { contacts: [], unassigned: [] }, c);
  assert.equal(detectedButDropped.detection.email.f1, 1);
  assert.equal(detectedButDropped.extractedEntities.email.fn, 1);
});

test("negative documents count invented entities and empty fields cannot pass", () => {
  const c = { name: "negative", input, entities: [], contacts: [], unassigned: [] };
  const result = score([entities[0]], { contacts: [{ person: entities[0] }], unassigned: [] }, c);
  assert.equal(result.detection.person.fp, 1);
  assert.equal(result.contactFields.person.fp, 1);
  assert.equal(result.exactCards.fp, 1);
  assert.equal(result.detection.address.passes95, false);
  assert.equal(result.allFieldsPass95, false);
});

test("normalization and region errors remain visible with perfect spans", () => {
  const c = { name: "values", input: "(202) 555-0199", entities: [{ kind: "phone", start: 0, end: 14, text: "(202) 555-0199", normalized: "+12025550199", region: "US" }] };
  const result = scoreCases([{ gold: c, detected: [{ ...c.entities[0], normalized: "+12025550198" }] }]);
  assert.equal(result.detection.phone.f1, 1);
  assert.equal(result.normalization.phone.fn, 1);
  const wrongRegion = scoreCases([{ gold: c, detected: [{ ...c.entities[0], region: "CA" }] }]);
  assert.equal(wrongRegion.normalization.phone.fn, 1);
});

test("address components require the correct detected address boundaries", () => {
  const address = { kind: "address", text: "500 Main St", start: 0, end: 11, components: [{ label: "house_number", text: "500", start: 0, end: 3 }, { label: "road", text: "Main St", start: 4, end: 11 }] };
  const c = { name: "address", input: address.text, entities: [address] };
  const actual = { ...address, components: [address.components[0]] };
  const result = scoreCases([{ gold: c, detected: [actual] }]);
  assert.equal(result.detection.address.f1, 1);
  assert.equal(result.addressComponents.fn, 1);
  assert.equal(result.completeDetectedAddressParses.tp, 0);
});

test("native byte offsets convert safely across emoji and combining marks", () => {
  const c = { name: "unicode", input: "😀 Café\u0301 Jane Doe", entities: [{ kind: "person", text: "Jane Doe", start: 13, end: 21 }] };
  const converted = fromUtf8Case(c);
  validateGold(converted);
  assert.equal(converted.entities[0].start, 9);
  assert.equal(converted.entities[0].end, 17);
  assert.throws(() => fromUtf8Case({ ...c, entities: [{ ...c.entities[0], start: 1 }] }), /character boundary/u);
});

test("invalid gold and missing outputs cannot silently pass", () => {
  assert.throws(() => validateGold({ ...gold, entities: [...entities, entities[0]] }), /duplicate gold/u);
  assert.throws(() => validateGold({ ...gold, unassigned: [0] }), /exactly one/u);
  assert.throws(() => scoreCases([{ gold, detected: entities }]), /missing extraction/u);
  assert.throws(() => scoreCases([]), /needs cases/u);
  assert.throws(() => scoreCases([{ gold, detected: entities, extraction }, { gold, detected: entities, extraction }]), /duplicate case/u);
});

test("gate uses unrounded precision rather than a displayed percentage", () => {
  const c = { name: "rounding", input: "x ".repeat(20001), entities: Array.from({ length: 19000 }, (_, i) => ({ kind: "person", text: "x", start: i * 2, end: i * 2 + 1 })) };
  const detected = [...c.entities, ...Array.from({ length: 1001 }, (_, i) => ({ kind: "person", text: "x", start: (19000 + i) * 2, end: (19000 + i) * 2 + 1 }))];
  const result = scoreCases([{ gold: c, detected }]);
  assert.equal((result.detection.person.precision * 100).toFixed(2), "95.00");
  assert.equal(result.detection.person.passes95, false);
});

function completeRow(name = "complete") {
  const text = "Jane Doe\nAcme\n500 Main St\njane@example.org\n(202) 555-0199";
  const at = (kind, value, metadata = {}) => ({ kind, text: value, start: text.indexOf(value), end: text.indexOf(value) + value.length, ...metadata });
  const address = at("address", "500 Main St", { components: [
    { label: "house_number", text: "500", start: text.indexOf("500"), end: text.indexOf("500") + 3 },
    { label: "road", text: "Main St", start: text.indexOf("Main St"), end: text.indexOf("Main St") + 7 },
  ] });
  const fields = [at("person", "Jane Doe"), at("org", "Acme"), address, at("email", "jane@example.org", { normalized: "jane@example.org" }), at("phone", "(202) 555-0199", { normalized: "+12025550199", region: "US" })];
  const c = { name, input: text, entities: fields, contacts: [{ person: 0, org: 1, addresses: [2], emails: [3], phones: [4] }], unassigned: [] };
  const card = { person: fields[0], org: fields[1], addresses: [fields[2]], emails: [fields[3]], phones: [fields[4]] };
  const output = { contacts: [card], unassigned: [] };
  return { gold: c, detected: fields, extraction: output };
}

test("complete five-field scores require correct values in contact output too", () => {
  const row = completeRow();
  const { gold: c, detected: fields, extraction: output } = row;
  const card = output.contacts[0];
  const correct = scoreCases([row]);
  assert.equal(correct.allFieldsPass95, true);
  const wrongValue = scoreCases([{ gold: c, detected: fields, extraction: { ...output, contacts: [{ ...card, phones: [{ ...fields[4], normalized: "+12025550198" }] }] } }]);
  assert.equal(wrongValue.allExactFieldsPass95, true);
  assert.equal(wrongValue.contactValues.phone.fn, 1);
  assert.equal(wrongValue.allFieldsPass95, false);
});


test("unassigned values are required even when contact fields are correct", () => {
  const row = completeRow();
  row.gold.input += "\n(303) 555-0100";
  const start = row.gold.input.indexOf("(303)");
  const phone = { kind: "phone", start, end: row.gold.input.length, text: "(303) 555-0100", normalized: "+13035550100", region: "US" };
  row.gold.entities.push(phone);
  row.gold.unassigned.push(5);
  row.detected = [...row.detected];
  row.extraction.unassigned.push({ ...phone, normalized: "+13035550101", region: "CA" });
  const result = scoreCases([row]);
  assert.equal(result.allExactFieldsPass95, true);
  assert.equal(result.normalization.phone.f1, 1);
  assert.equal(result.extractedValues.phone.fn, 1);
  assert.equal(result.allFieldsPass95, false);
});

test("missing unassigned address component labels prevent a complete gate", () => {
  const row = completeRow();
  row.gold.input += "\n600 Oak St";
  const start = row.gold.input.indexOf("600");
  const address = { kind: "address", start, end: row.gold.input.length, text: "600 Oak St" };
  row.gold.entities.push(address);
  row.gold.unassigned.push(5);
  row.detected = [...row.detected];
  row.extraction.unassigned.push(address);
  const result = scoreCases([row]);
  assert.equal(result.allExactFieldsPass95, true);
  assert.equal(result.coverage.addressComponentCases, 0);
  assert.equal(result.allFieldsPass95, false);
});

test("phone region must be explicitly labeled", () => {
  const row = completeRow();
  const unlabeled = { ...row.gold.entities[4] };
  delete unlabeled.region;
  row.gold.entities[4] = unlabeled;
  const result = scoreCases([row]);
  assert.equal(result.allExactFieldsPass95, true);
  assert.equal(result.coverage.normalizationCases.phone, 0);
  assert.equal(result.allFieldsPass95, false);
});

test("address negatives contribute false components and complete parses", () => {
  const positives = Array.from({ length: 100 }, (_, i) => completeRow(`positive-${i}`));
  for (const row of positives.slice(0, 5)) {
    row.detected = row.detected.map((e) => e.kind === "address" ? { ...e, components: [] } : e);
  }
  const negatives = Array.from({ length: 5 }, (_, i) => {
    const address = completeRow().detected[2];
    return { gold: { name: `negative-${i}`, input: completeRow().gold.input, entities: [], contacts: [], unassigned: [] }, detected: [address], extraction: { contacts: [], unassigned: [address] } };
  });
  const result = scoreCases([...positives, ...negatives]);
  assert.equal(result.detection.address.passes95, true);
  assert.equal(result.coverage.addressComponentCases, 105);
  assert.equal(result.completeDetectedAddressParses.predicted, 105);
  assert.equal(result.completeDetectedAddressParses.precision, 95 / 105);
  assert.equal(result.addressComponents.fp, 10);
  assert.equal(result.allFieldsPass95, false);
});

test("null components cannot masquerade as a reviewed empty parse", () => {
  const row = completeRow();
  row.gold.entities[2] = { ...row.gold.entities[2], components: null };
  assert.throws(() => scoreCases([row]), /components must be an array/u);
  const missing = completeRow();
  missing.detected = missing.detected.map((e) => e.kind === "address" ? { ...e, components: null } : e);
  assert.throws(() => scoreCases([missing]), /components must be an array/u);
});
