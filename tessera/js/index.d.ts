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

export type ErrorCode =
  | "BUNDLE_INVALID"
  | "CHECKSUM_MISMATCH"
  | "UNSUPPORTED_VERSION"
  | "INPUT_TOO_LARGE"
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
   * not start a worker, as for a package loaded cross-origin from a CDN, calls run inline.
   */
  worker?: boolean;
}

export interface QueryOptions {
  /** Regions for phone numbers written without a country code. Ignored by `parseAddress`. */
  countryHint?: string[];
  /** Return low-confidence results instead of omitting them. */
  includeUncertain?: boolean;
}

export interface Tessera {
  /** The kinds this instance serves, in taxonomy order; empty once disposed. */
  readonly kinds: Kind[];
  /** Every supported entity in `text`, in document order. */
  detect(text: string, options?: QueryOptions): Promise<Entity[]>;
  /** Split `text`, known to be one address, into components. */
  parseAddress(text: string, options?: QueryOptions): Promise<Entity>;
  /** Release the model and any worker; later calls reject with `DISPOSED`. */
  dispose(): void;
}

export function createTessera(options?: CreateOptions): Promise<Tessera>;
