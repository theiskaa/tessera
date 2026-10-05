const KINDS = ["person", "org", "address", "email", "phone"];
const RULE_KINDS = ["email", "phone"];
const ADDRESS_LABELS = ["house_number", "road", "unit", "level", "suburb", "city", "district", "region", "postcode", "country", "po_box", "unknown"];
const own = (value, key) => Object.hasOwn(value, key);
const spanKey = (e) => JSON.stringify([e.kind, e.start, e.end]);
const componentKey = (address, c) => JSON.stringify([spanKey(address), c.label, c.start, c.end]);
const members = (c) => [c.person, c.org, ...(c.addresses ?? []), ...(c.emails ?? []), ...(c.phones ?? [])].filter(Boolean);

function require(condition, message) {
  if (!condition) throw new Error(message);
}

function validateSpan(input, e, label = "kind") {
  require(Number.isSafeInteger(e.start) && Number.isSafeInteger(e.end) && e.start >= 0 && e.end > e.start && e.end <= input.length, "invalid span offsets");
  require(input.slice(e.start, e.end) === e.text, "span does not slice to its text");
  require(label === "kind" ? KINDS.includes(e.kind) : ADDRESS_LABELS.includes(e.label), "invalid span label");
  require(!own(e, "components") || e.kind === "address" && Array.isArray(e.components), "address components must be an array");
  for (const c of e.components ?? []) {
    validateSpan(input, c, "label");
    require(c.start >= e.start && c.end <= e.end, "component outside its address");
  }
}

/** Convert native fixture byte offsets to the public JavaScript UTF-16 contract. */
export function fromUtf8Case(c) {
  const offsets = new Map([[0, 0]]);
  let bytes = 0;
  let units = 0;
  for (const char of c.input) {
    bytes += new TextEncoder().encode(char).length;
    units += char.length;
    offsets.set(bytes, units);
  }
  const convert = (e) => {
    require(offsets.has(e.start) && offsets.has(e.end), "UTF-8 offset is not a character boundary");
    return { ...e, start: offsets.get(e.start), end: offsets.get(e.end), ...(e.components ? { components: e.components.map(convert) } : {}) };
  };
  return { ...c, entities: (c.entities ?? c.expected).map(convert) };
}

/** Validate complete entity and optional contact labels before evaluating any predictions. */
export function validateGold(c) {
  require(typeof c.name === "string" && c.name.length > 0 && typeof c.input === "string", "case needs a name and input");
  require(Array.isArray(c.entities), "case needs entity labels");
  const spans = new Set();
  for (const e of c.entities) {
    validateSpan(c.input, e);
    require(!spans.has(spanKey(e)), "duplicate gold entity");
    spans.add(spanKey(e));
    require(!own(e, "normalized") || typeof e.normalized === "string" && e.normalized.length > 0, "invalid normalized label");
    require(!own(e, "region") || e.region === null || typeof e.region === "string" && e.region.length > 0, "invalid region label");
    if (own(e, "components")) {
      const keys = e.components.map((part) => componentKey(e, part));
      require(new Set(keys).size === keys.length, "duplicate gold component");
    }
  }
  if (c.contacts === undefined) return;
  require(Array.isArray(c.contacts) && Array.isArray(c.unassigned), "contact labels need contacts and unassigned arrays");
  const ownership = c.entities.map(() => 0);
  const anchors = new Set();
  for (const contact of c.contacts) {
    const anchor = contact.person ?? contact.org;
    require(Number.isSafeInteger(anchor), "contact needs an anchor");
    require(!anchors.has(anchor), "duplicate gold contact anchor");
    anchors.add(anchor);
    for (const [kind, indices] of contactIndices(contact)) {
      for (const index of indices) {
        require(Number.isSafeInteger(index) && c.entities[index]?.kind === kind, "invalid contact member");
        ownership[index] += 1;
      }
    }
  }
  for (const index of c.unassigned) {
    require(Number.isSafeInteger(index) && index >= 0 && index < ownership.length, "invalid unassigned entity");
    ownership[index] += 1;
  }
  require(ownership.every((count) => count === 1), "each gold entity needs exactly one owner or unassigned label");
}

function contactIndices(c) {
  return [["person", c.person == null ? [] : [c.person]], ["org", c.org == null ? [] : [c.org]], ["address", c.addresses ?? []], ["email", c.emails ?? []], ["phone", c.phones ?? []]];
}

function matchCount(gold, predicted) {
  const remaining = new Map();
  for (const key of gold) remaining.set(key, (remaining.get(key) ?? 0) + 1);
  let tp = 0;
  for (const key of predicted) {
    if ((remaining.get(key) ?? 0) > 0) {
      tp += 1;
      remaining.set(key, remaining.get(key) - 1);
    }
  }
  return { tp, predicted: predicted.length, gold: gold.length };
}

function emptyKinds() {
  return Object.fromEntries(KINDS.map((kind) => [kind, { tp: 0, predicted: 0, gold: 0 }]));
}

function add(target, counts) {
  for (const key of ["tp", "predicted", "gold"]) target[key] += counts[key];
}

function metrics(c) {
  const precision = c.predicted ? c.tp / c.predicted : 0;
  const recall = c.gold ? c.tp / c.gold : 0;
  return { ...c, fp: c.predicted - c.tp, fn: c.gold - c.tp, precision, recall, f1: c.predicted + c.gold ? 2 * c.tp / (c.predicted + c.gold) : 0, passes95: c.gold > 0 && c.predicted > 0 && precision >= 0.95 && recall >= 0.95 };
}

function goldFields(c) {
  return c.contacts.flatMap((contact) => {
    const owner = spanKey(c.entities[contact.person ?? contact.org]);
    return contactIndices(contact).flatMap(([kind, indices]) => indices.map((i) => ({ kind, entity: c.entities[i], key: JSON.stringify([owner, spanKey(c.entities[i])]) })));
  });
}

function predictedFields(extraction) {
  return extraction.contacts.flatMap((contact) => {
    const anchor = contact.person ?? contact.org;
    const owner = anchor ? spanKey(anchor) : null;
    return members(contact).map((e) => ({ kind: e.kind, entity: e, key: JSON.stringify([owner, spanKey(e)]) }));
  });
}

function hasValueLabel(e) {
  if (e.kind === "phone") return own(e, "normalized") && own(e, "region");
  return e.kind === "email" ? own(e, "normalized") : e.kind !== "address" || own(e, "components");
}

function valueKey(field, expectedBySpan) {
  const e = field.entity;
  const gold = expectedBySpan.get(spanKey(e));
  return JSON.stringify([
    field.key,
    RULE_KINDS.includes(e.kind) ? e.normalized ?? null : null,
    gold && own(gold, "region") ? e.region ?? null : null,
    e.kind === "address" ? (e.components ?? []).map((part) => componentKey(e, part)).sort() : null,
  ]);
}

/** Score public API outputs; the result is diagnostic evidence, not release authorization. */
export function scoreCases(rows) {
  require(Array.isArray(rows) && rows.length > 0, "evaluation needs cases");
  const detection = emptyKinds();
  const contactFields = emptyKinds();
  const extracted = emptyKinds();
  const contactValues = emptyKinds();
  const detectionValues = emptyKinds();
  const extractedValues = emptyKinds();
  const normalization = emptyKinds();
  const components = { tp: 0, predicted: 0, gold: 0 };
  const addressParses = { tp: 0, predicted: 0, gold: 0 };
  const exactCards = { tp: 0, predicted: 0, gold: 0 };
  const normalizationCases = { email: 0, phone: 0 };
  const contactValueCases = Object.fromEntries(KINDS.map((kind) => [kind, 0]));
  const detectionValueCases = Object.fromEntries(KINDS.map((kind) => [kind, 0]));
  const extractedValueCases = Object.fromEntries(KINDS.map((kind) => [kind, 0]));
  const falsePositiveDocuments = Object.fromEntries(KINDS.map((kind) => [kind, 0]));
  let contactCases = 0;
  let addressComponentCases = 0;
  const names = new Set();
  for (const { gold: c, detected, extraction } of rows) {
    validateGold(c);
    require(!names.has(c.name), "duplicate case name");
    names.add(c.name);
    require(Array.isArray(detected), "missing detection output");
    for (const e of detected) validateSpan(c.input, e);
    const expectedBySpan = new Map(c.entities.map((e) => [spanKey(e), e]));
    const entityValues = (entities) => entities.map((entity) => valueKey({ entity, key: spanKey(entity) }, expectedBySpan));
    for (const kind of KINDS) {
      const expected = c.entities.filter((e) => e.kind === kind);
      const actual = detected.filter((e) => e.kind === kind);
      const counts = matchCount(expected.map(spanKey), actual.map(spanKey));
      add(detection[kind], counts);
      if (expected.every(hasValueLabel)) {
        detectionValueCases[kind] += 1;
        add(detectionValues[kind], matchCount(entityValues(expected), entityValues(actual)));
      }
      if (counts.predicted > counts.tp) falsePositiveDocuments[kind] += 1;
      if (RULE_KINDS.includes(kind) && expected.every(hasValueLabel)) {
        normalizationCases[kind] += 1;
        const valueKey = (e) => JSON.stringify([spanKey(e), e.normalized]);
        const regionKeys = expected.map((e) => JSON.stringify([valueKey(e), own(e, "region") ? e.region : null]));
        const actualKeys = actual.map((e) => {
          const expectedEntity = expected.find((g) => spanKey(g) === spanKey(e));
          return JSON.stringify([valueKey(e), expectedEntity && own(expectedEntity, "region") ? e.region ?? null : null]);
        });
        add(normalization[kind], matchCount(regionKeys, actualKeys));
      }
    }
    const expectedAddresses = c.entities.filter((e) => e.kind === "address");
    if (expectedAddresses.every((e) => own(e, "components"))) {
      addressComponentCases += 1;
      const actualAddresses = detected.filter((e) => e.kind === "address");
      const parts = (entities) => entities.flatMap((e) => (e.components ?? []).map((part) => componentKey(e, part)));
      add(components, matchCount(parts(expectedAddresses), parts(actualAddresses)));
      const parseKey = (e) => JSON.stringify([spanKey(e), (e.components ?? []).map((part) => componentKey(e, part)).sort()]);
      add(addressParses, matchCount(expectedAddresses.map(parseKey), actualAddresses.map(parseKey)));
    }
    if (c.contacts === undefined) continue;
    contactCases += 1;
    require(Array.isArray(extraction?.contacts) && Array.isArray(extraction?.unassigned), "missing extraction output");
    const extractedEntities = [...extraction.contacts.flatMap(members), ...extraction.unassigned];
    for (const e of extractedEntities) validateSpan(c.input, e);
    const expected = goldFields(c);
    const actual = predictedFields(extraction);
    for (const kind of KINDS) {
      const expectedKind = expected.filter((e) => e.kind === kind);
      const actualKind = actual.filter((e) => e.kind === kind);
      add(extracted[kind], matchCount(c.entities.filter((e) => e.kind === kind).map(spanKey), extractedEntities.filter((e) => e.kind === kind).map(spanKey)));
      const expectedEntities = c.entities.filter((e) => e.kind === kind);
      if (expectedEntities.every(hasValueLabel)) {
        extractedValueCases[kind] += 1;
        add(extractedValues[kind], matchCount(entityValues(expectedEntities), entityValues(extractedEntities.filter((e) => e.kind === kind))));
      }
      add(contactFields[kind], matchCount(expectedKind.map((e) => e.key), actualKind.map((e) => e.key)));
      if (expectedKind.every((field) => hasValueLabel(field.entity))) {
        contactValueCases[kind] += 1;
        add(contactValues[kind], matchCount(expectedKind.map((field) => valueKey(field, expectedBySpan)), actualKind.map((field) => valueKey(field, expectedBySpan))));
      }
    }
    const expectedCards = c.contacts.map((contact) => JSON.stringify(contactIndices(contact).flatMap(([, indices]) => indices.map((i) => spanKey(c.entities[i]))).sort()));
    const actualCards = extraction.contacts.map((contact) => JSON.stringify(members(contact).map(spanKey).sort()));
    add(exactCards, matchCount(expectedCards, actualCards));
  }
  const summarize = (counts) => Object.fromEntries(Object.entries(counts).map(([kind, c]) => [kind, metrics(c)]));
  const result = {
    metricVersion: 1, scope: "diagnostic; no independent release claim", cases: rows.length,
    coverage: { contactCases, normalizationCases, addressComponentCases, contactValueCases, detectionValueCases, extractedValueCases },
    detection: summarize(detection), extractedEntities: summarize(extracted), contactFields: summarize(contactFields), contactValues: summarize(contactValues), detectionValues: summarize(detectionValues), extractedValues: summarize(extractedValues), normalization: summarize(normalization),
    addressComponents: metrics(components), completeDetectedAddressParses: metrics(addressParses), exactCards: metrics(exactCards),
    exactFalsePositiveDocumentRate: Object.fromEntries(KINDS.map((kind) => [kind, falsePositiveDocuments[kind] / rows.length])),
  };
  result.allExactFieldsPass95 = contactCases === rows.length && [result.detection, result.extractedEntities, result.contactFields].every((scores) => KINDS.every((kind) => scores[kind].passes95));
  result.allFieldsPass95 = result.allExactFieldsPass95 && KINDS.every((kind) => contactValueCases[kind] === contactCases && result.contactValues[kind].passes95)
    && KINDS.every((kind) => detectionValueCases[kind] === rows.length && result.detectionValues[kind].passes95 && extractedValueCases[kind] === contactCases && result.extractedValues[kind].passes95)
    && RULE_KINDS.every((kind) => normalizationCases[kind] === rows.length && result.normalization[kind].passes95)
    && addressComponentCases === rows.length && result.completeDetectedAddressParses.passes95 && result.addressComponents.passes95;
  return result;
}
