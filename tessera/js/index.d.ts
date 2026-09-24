/** "person" and "org" are experimental: reliable only on documents shaped like the detector's training data. */
export type Kind = "person" | "org" | "address" | "email" | "phone";

export type Source = "model" | "rules";

export type AddressLabel =
  | "house_number"
  | "road"
  | "unit"
  | "level"
  | "suburb"
  | "city"
  | "district"
  | "region"
  | "postcode"
  | "country"
  | "po_box"
  | "unknown";

/** Offsets are UTF-16 code units into the source string; `end` is exclusive. */
export interface Component {
  label: AddressLabel;
  text: string;
  start: number;
  end: number;
  confidence: number;
}

export interface Entity {
  kind: Kind;
  text: string;
  start: number;
  end: number;
  confidence: number;
  source: Source;
  reviewRecommended: boolean;
  /** Present on address entities only. */
  components?: Component[];
  /** E.164 for phones; for emails the address with its domain lowercased. */
  normalized?: string;
  /** Phone only, ISO 3166-1 alpha-2. */
  region?: string;
}

/** Entities that belong to one person or organization. Offsets are UTF-16 code units. */
export interface Contact {
  /** First unit of any of the contact's entities. */
  start: number;
  /** End of the last of them, exclusive. */
  end: number;
  /** The weakest assignment's confidence, never an average. */
  confidence: number;
  reviewRecommended: boolean;
  /** The anchor, when the contact is a person's. */
  person?: Entity;
  /** The anchor when there is no person, else the person's organization. */
  org?: Entity;
  addresses: Entity[];
  emails: Entity[];
  phones: Entity[];
}

export interface Extraction {
  /** Contacts in document order. */
  contacts: Contact[];
  /** Entities no contact could take with confidence, in document order. */
  unassigned: Entity[];
}

export type ErrorCode =
  | "BUNDLE_INVALID"
  | "CHECKSUM_MISMATCH"
  | "UNSUPPORTED_VERSION"
  | "INPUT_TOO_LARGE"
  | "UNSUPPORTED_FORMAT"
  | "INFERENCE"
  | "MODEL_FETCH_FAILED"
  | "UNSUPPORTED_RUNTIME"
  | "DISPOSED";

/** Every library failure. Malformed arguments are plain `TypeError`s instead. */
export interface TesseraError extends Error {
  name: "TesseraError";
  code: ErrorCode;
  /** The failing pipeline stage, for `INFERENCE`. */
  stage?: string;
}

export interface CreateOptions {
  /** URL of the weight bundle. Not needed when `kinds` contains only "email" and "phone". */
  modelUrl?: string;
  /** Bundle bytes, used instead of fetching `modelUrl`. */
  modelBytes?: Uint8Array | ArrayBuffer;
  /** Integrity string the bundle must match, `sha256-…`. */
  integrity?: string;
  /** Kinds to detect, at least one. Defaults to all. */
  kinds?: Kind[];
  /**
   * Run inference in a Web Worker on a browser main thread; ignored elsewhere. Where the page may
   * not start a worker, because the `Worker` constructor throws as it does for a package loaded
   * cross-origin from a CDN, calls run inline. A worker that starts but whose script or module
   * fails to load rejects with `UNSUPPORTED_RUNTIME` instead.
   */
  worker?: boolean;
}

export interface MarkdownOptions {
  /** Scan fenced, indented, and inline code. Default false. */
  includeCode?: boolean;
  /** Scan raw HTML blocks and inline HTML as text. Default false. */
  includeHtml?: boolean;
  /** Parse GFM tables so each cell is scanned separately. Default true. */
  gfmTables?: boolean;
}

export interface QueryOptions {
  /** Regions for phone numbers written without a country code. Ignored by `parseAddress`. */
  countryHint?: string[];
  /** Return low-confidence results instead of omitting them. */
  includeUncertain?: boolean;
  /**
   * Default "text". "markdown" scans only prose and reads `mailto:` and `tel:` link destinations;
   * offsets still index the Markdown source. A build without Markdown support rejects it with
   * `UNSUPPORTED_FORMAT`. Ignored by `parseAddress`.
   */
  format?: "text" | "markdown";
  /** How "markdown" input is selected; ignored for "text". */
  markdown?: MarkdownOptions;
}

/**
 * A loaded extractor. An instance nothing references any more is freed by the garbage collector;
 * a worker-backed one first answers the calls already made, then stops its worker.
 */
export interface Tessera {
  /** The kinds this instance serves, in taxonomy order; empty once disposed. */
  readonly kinds: Kind[];
  /** Every supported entity in `text`, in document order. */
  detect(text: string, options?: QueryOptions): Promise<Entity[]>;
  /**
   * The entities of `detect`, grouped into contacts. A wrong assignment is worse than none, so
   * whatever no rule places confidently is in `unassigned`.
   */
  extractContacts(text: string, options?: QueryOptions): Promise<Extraction>;
  /** Split `text`, known to be one address, into components. */
  parseAddress(text: string, options?: QueryOptions): Promise<Entity>;
  /**
   * Release the model and any worker. Calls still waiting on the worker and every later call
   * reject with `DISPOSED`; disposing twice is harmless.
   */
  dispose(): void;
  /**
   * Free the instance's memory. A worker-backed instance answers the calls already made before
   * its worker stops. Afterwards the object is unusable: every member throws synchronously
   * instead of rejecting with `DISPOSED`, so prefer `dispose()`.
   */
  free(): void;
}

export function createTessera(options?: CreateOptions): Promise<Tessera>;
