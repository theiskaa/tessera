//! Public bodies and named units for the org slots: government departments, agencies,
//! courts, councils, and their sub-units, with acronyms where they have them. The name pools
//! are almost all companies, while real documents name public bodies far more often, and
//! refer to them by acronym after the first mention.
//!
//! Listed bodies are real; composed ones join a pattern with a topic or a city, so the model
//! sees shapes, not one fixed list. Both are split by a hash of the name, like the name pools,
//! so the synthetic test split holds bodies the model never trained on.

use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;
use tessera::internal::fnv1a;

use crate::data::Split;

/// A public body or unit and its acronym, if it has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Body {
    pub name: String,
    pub acronym: Option<String>,
}

type Listed = &'static [(&'static str, &'static str)];

/// Real bodies per country, `""` where a body has no acronym in use. `EU` is mixed into every
/// country's draws.
const LISTED: &[(&str, Listed)] = &[
    (
        "US",
        &[
            ("Environmental Protection Agency", "EPA"),
            ("Food and Drug Administration", "FDA"),
            ("Federal Communications Commission", "FCC"),
            ("Securities and Exchange Commission", "SEC"),
            ("Federal Trade Commission", "FTC"),
            ("Federal Deposit Insurance Corporation", "FDIC"),
            ("Office of Management and Budget", "OMB"),
            ("National Oceanic and Atmospheric Administration", "NOAA"),
            ("National Institutes of Health", "NIH"),
            ("Centers for Disease Control and Prevention", "CDC"),
            ("Internal Revenue Service", "IRS"),
            ("Social Security Administration", "SSA"),
            ("Department of Energy", "DOE"),
            ("Department of Transportation", "DOT"),
            ("Department of Labor", "DOL"),
            ("Department of Agriculture", "USDA"),
            ("Department of Veterans Affairs", "VA"),
            ("Department of Homeland Security", "DHS"),
            ("Department of Justice", "DOJ"),
            ("Department of the Treasury", ""),
            ("Department of Housing and Urban Development", "HUD"),
            ("Federal Energy Regulatory Commission", "FERC"),
            ("Nuclear Regulatory Commission", "NRC"),
            ("Occupational Safety and Health Administration", "OSHA"),
            ("National Marine Fisheries Service", "NMFS"),
            ("Fish and Wildlife Service", "FWS"),
            ("Bureau of Land Management", "BLM"),
            ("Federal Aviation Administration", "FAA"),
            ("Federal Highway Administration", "FHWA"),
            ("Federal Railroad Administration", "FRA"),
            ("Consumer Product Safety Commission", "CPSC"),
            ("Consumer Financial Protection Bureau", "CFPB"),
            ("General Services Administration", "GSA"),
            ("Small Business Administration", "SBA"),
            ("National Science Foundation", "NSF"),
            ("National Institute of Standards and Technology", "NIST"),
            ("Census Bureau", ""),
            ("Bureau of Labor Statistics", "BLS"),
            ("Health Resources and Services Administration", "HRSA"),
            ("Centers for Medicare & Medicaid Services", "CMS"),
            ("Commodity Futures Trading Commission", "CFTC"),
            ("Equal Employment Opportunity Commission", "EEOC"),
            ("National Labor Relations Board", "NLRB"),
            ("United States Postal Service", "USPS"),
            ("International Trade Administration", "ITA"),
            ("United States Patent and Trademark Office", "USPTO"),
            ("U.S. Army Corps of Engineers", "USACE"),
            ("National Park Service", "NPS"),
            ("Federal Emergency Management Agency", "FEMA"),
            ("Transportation Security Administration", "TSA"),
            ("U.S. Customs and Border Protection", "CBP"),
            ("Drug Enforcement Administration", "DEA"),
            ("Federal Motor Carrier Safety Administration", "FMCSA"),
            (
                "Pipeline and Hazardous Materials Safety Administration",
                "PHMSA",
            ),
            ("Bureau of Ocean Energy Management", "BOEM"),
            ("Administration for Children and Families", "ACF"),
            (
                "Substance Abuse and Mental Health Services Administration",
                "SAMHSA",
            ),
            ("Federal Housing Finance Agency", "FHFA"),
            ("Office of Personnel Management", "OPM"),
            ("National Archives and Records Administration", "NARA"),
        ],
    ),
    (
        "GB",
        &[
            ("HM Revenue & Customs", "HMRC"),
            ("Driver and Vehicle Licensing Agency", "DVLA"),
            ("Department for Work and Pensions", "DWP"),
            ("Foreign, Commonwealth & Development Office", "FCDO"),
            ("Home Office", ""),
            ("Ministry of Justice", "MOJ"),
            ("Ministry of Defence", "MOD"),
            ("Department for Education", "DfE"),
            ("Department of Health and Social Care", "DHSC"),
            ("Department for Transport", "DfT"),
            ("Department for Environment, Food & Rural Affairs", "Defra"),
            ("Department for Business and Trade", "DBT"),
            ("Cabinet Office", ""),
            ("HM Treasury", ""),
            ("Companies House", ""),
            ("Environment Agency", ""),
            ("Maritime and Coastguard Agency", "MCA"),
            ("Financial Conduct Authority", "FCA"),
            ("Prudential Regulation Authority", "PRA"),
            ("Information Commissioner's Office", "ICO"),
            ("Competition and Markets Authority", "CMA"),
            ("Office of Communications", "Ofcom"),
            ("Office of Gas and Electricity Markets", "Ofgem"),
            ("Office for National Statistics", "ONS"),
            ("Health and Safety Executive", "HSE"),
            ("Care Quality Commission", "CQC"),
            ("Charity Commission", ""),
            ("HM Land Registry", ""),
            ("Planning Inspectorate", ""),
            ("Met Office", ""),
            ("NHS England", ""),
            ("UK Health Security Agency", "UKHSA"),
            ("Intellectual Property Office", "IPO"),
            ("Insolvency Service", ""),
            ("Student Loans Company", "SLC"),
            ("Crown Prosecution Service", "CPS"),
            (
                "Ministry of Housing, Communities and Local Government",
                "MHCLG",
            ),
            ("Scottish Government", ""),
            ("Welsh Government", ""),
            ("Bank of England", ""),
            ("Serious Fraud Office", "SFO"),
            ("Disclosure and Barring Service", "DBS"),
            ("Valuation Office Agency", "VOA"),
            ("Driver and Vehicle Standards Agency", "DVSA"),
            ("UK Visas and Immigration", "UKVI"),
            ("Office of Rail and Road", "ORR"),
            ("Pensions Regulator", "TPR"),
            ("Food Standards Agency", "FSA"),
            (
                "Medicines and Healthcare products Regulatory Agency",
                "MHRA",
            ),
            ("Department for Energy Security and Net Zero", "DESNZ"),
        ],
    ),
    (
        "DE",
        &[
            ("Bundesministerium der Finanzen", "BMF"),
            ("Bundesministerium des Innern", "BMI"),
            ("Bundesministerium für Wirtschaft und Klimaschutz", "BMWK"),
            ("Bundesministerium für Gesundheit", "BMG"),
            ("Bundesministerium der Justiz", "BMJ"),
            ("Bundesministerium für Arbeit und Soziales", "BMAS"),
            ("Bundesministerium für Bildung und Forschung", "BMBF"),
            ("Bundesministerium für Digitales und Verkehr", "BMDV"),
            ("Auswärtiges Amt", ""),
            ("Bundesamt für Migration und Flüchtlinge", "BAMF"),
            ("Bundesamt für Sicherheit in der Informationstechnik", "BSI"),
            ("Bundeszentralamt für Steuern", "BZSt"),
            ("Bundesagentur für Arbeit", ""),
            ("Bundesnetzagentur", "BNetzA"),
            ("Bundeskartellamt", ""),
            ("Bundesanstalt für Finanzdienstleistungsaufsicht", "BaFin"),
            ("Deutsche Bundesbank", ""),
            ("Statistisches Bundesamt", "Destatis"),
            ("Kraftfahrt-Bundesamt", "KBA"),
            ("Umweltbundesamt", "UBA"),
            ("Deutsches Patent- und Markenamt", "DPMA"),
            ("Bundesverfassungsgericht", "BVerfG"),
            ("Bundesgerichtshof", "BGH"),
            ("Bundesfinanzhof", "BFH"),
            ("Deutsche Rentenversicherung Bund", "DRV"),
            (
                "Bundesinstitut für Arzneimittel und Medizinprodukte",
                "BfArM",
            ),
            ("Bundesamt für Justiz", "BfJ"),
            ("Informationstechnikzentrum Bund", "ITZBund"),
            ("Bundesamt für Verfassungsschutz", "BfV"),
            ("Bundesanstalt für Straßenwesen", "BASt"),
            ("Eisenbahn-Bundesamt", "EBA"),
            ("Bundesamt für Wirtschaft und Ausfuhrkontrolle", "BAFA"),
        ],
    ),
    (
        "GE",
        &[
            ("საქართველოს ფინანსთა სამინისტრო", ""),
            ("საქართველოს იუსტიციის სამინისტრო", ""),
            ("საქართველოს შინაგან საქმეთა სამინისტრო", ""),
            ("საქართველოს საგარეო საქმეთა სამინისტრო", ""),
            (
                "საქართველოს ეკონომიკისა და მდგრადი განვითარების სამინისტრო",
                "",
            ),
            (
                "საქართველოს გარემოს დაცვისა და სოფლის მეურნეობის სამინისტრო",
                "",
            ),
            ("შემოსავლების სამსახური", ""),
            ("საჯარო რეესტრის ეროვნული სააგენტო", ""),
            ("სახელმწიფო სერვისების განვითარების სააგენტო", ""),
            ("იუსტიციის სახლი", ""),
            ("საქართველოს ეროვნული ბანკი", ""),
            ("საქართველოს სტატისტიკის ეროვნული სამსახური", ""),
            ("თბილისის მერია", ""),
            ("ბათუმის მერია", ""),
            ("ქუთაისის მერია", ""),
            ("რუსთავის მერია", ""),
            ("თბილისის საქალაქო სასამართლო", ""),
            ("საქართველოს უზენაესი სასამართლო", ""),
            ("საქართველოს პარლამენტი", ""),
            ("კომუნიკაციების ეროვნული კომისია", ""),
            ("სახელმწიფო აუდიტის სამსახური", ""),
            ("ფინანსური მონიტორინგის სამსახური", ""),
            ("Ministry of Finance of Georgia", ""),
            ("Revenue Service of Georgia", ""),
            ("National Bank of Georgia", "NBG"),
            ("National Agency of Public Registry", "NAPR"),
            ("Public Service Hall", ""),
            ("Georgian National Communications Commission", "GNCC"),
            ("Tbilisi City Hall", ""),
            ("Parliament of Georgia", ""),
            ("Supreme Court of Georgia", ""),
            ("State Audit Office of Georgia", "SAO"),
            ("National Statistics Office of Georgia", "Geostat"),
            ("Ministry of Internal Affairs of Georgia", "MIA"),
            ("Ministry of Foreign Affairs of Georgia", "MFA"),
            (
                "Georgian National Energy and Water Supply Regulatory Commission",
                "GNERC",
            ),
            ("Financial Monitoring Service of Georgia", "FMS"),
            ("Tbilisi City Court", ""),
        ],
    ),
    (
        "JP",
        &[
            ("総務省", ""),
            ("法務省", ""),
            ("外務省", ""),
            ("財務省", ""),
            ("文部科学省", ""),
            ("厚生労働省", ""),
            ("農林水産省", ""),
            ("経済産業省", ""),
            ("国土交通省", ""),
            ("環境省", ""),
            ("内閣府", ""),
            ("デジタル庁", ""),
            ("金融庁", ""),
            ("消費者庁", ""),
            ("国税庁", ""),
            ("特許庁", ""),
            ("気象庁", ""),
            ("公正取引委員会", ""),
            ("個人情報保護委員会", ""),
            ("日本銀行", ""),
            ("日本年金機構", ""),
            ("東京都庁", ""),
            ("最高裁判所", ""),
            ("Ministry of Economy, Trade and Industry", "METI"),
            (
                "Ministry of Land, Infrastructure, Transport and Tourism",
                "MLIT",
            ),
            ("Ministry of Health, Labour and Welfare", "MHLW"),
            ("Ministry of Foreign Affairs of Japan", "MOFA"),
            ("Ministry of Internal Affairs and Communications", "MIC"),
            ("Financial Services Agency", "FSA"),
            ("Japan Patent Office", "JPO"),
            ("Bank of Japan", "BOJ"),
            ("Japan External Trade Organization", "JETRO"),
            ("Japan International Cooperation Agency", "JICA"),
            ("Personal Information Protection Commission", "PPC"),
            ("Japan Fair Trade Commission", "JFTC"),
            ("National Tax Agency", "NTA"),
            ("Digital Agency", ""),
            ("Consumer Affairs Agency", "CAA"),
            ("Japan Meteorological Agency", "JMA"),
            ("Tokyo Metropolitan Government", "TMG"),
        ],
    ),
    (
        "EU",
        &[
            ("European Commission", ""),
            ("European Parliament", ""),
            ("Council of the European Union", ""),
            ("European Central Bank", "ECB"),
            ("European Court of Auditors", "ECA"),
            ("Court of Justice of the European Union", "CJEU"),
            ("European Investment Bank", "EIB"),
            ("European Data Protection Supervisor", "EDPS"),
            ("European Food Safety Authority", "EFSA"),
            ("European Medicines Agency", "EMA"),
            ("European Union Intellectual Property Office", "EUIPO"),
            ("European Anti-Fraud Office", "OLAF"),
            ("European External Action Service", "EEAS"),
            ("Directorate-General for Energy", "DG ENER"),
            ("Directorate-General for Mobility and Transport", "DG MOVE"),
            ("Directorate-General for Competition", "DG COMP"),
            ("Directorate-General for Trade", "DG TRADE"),
            ("Directorate-General for Health and Food Safety", "DG SANTE"),
            ("European Economic and Social Committee", "EESC"),
            ("European Committee of the Regions", "CoR"),
            ("European Chemicals Agency", "ECHA"),
            ("European Banking Authority", "EBA"),
            ("European Securities and Markets Authority", "ESMA"),
            ("European Environment Agency", "EEA"),
            ("European Union Agency for Cybersecurity", "ENISA"),
            ("European Ombudsman", ""),
        ],
    ),
];

/// Topics that composed English bodies and units are about.
const TOPICS: &[&str] = &[
    "Pesticide Programs",
    "Fisheries Science",
    "Air Quality",
    "Consumer Protection",
    "Enforcement",
    "Market Oversight",
    "Grants Management",
    "Policy and Planning",
    "Energy Efficiency",
    "Road Safety",
    "Reactor Regulation",
    "Trade Remedies",
    "Food Safety",
    "Maritime Affairs",
    "Digital Services",
    "Research and Innovation",
    "Financial Stability",
    "Tax Policy",
    "Veterinary Medicine",
    "Coastal Management",
    "Rail Safety",
    "Public Health Preparedness",
    "International Affairs",
    "Civil Rights",
    "Workforce Development",
    "Clinical Research",
    "Border Security",
    "Economic Analysis",
    "Climate Adaptation",
    "Water Resources",
    "Housing Standards",
    "Aviation Safety",
    "Export Controls",
    "Data Protection",
    "Licensing and Registration",
    "Rural Development",
    "Wildlife Conservation",
    "Emergency Management",
    "Statistical Methods",
    "Infrastructure Investment",
];

/// Topics of named engineering and product teams, as signatures write them.
const TEAM_TOPICS: &[&str] = &[
    "Kernel Networking",
    "Platform Security",
    "Storage Drivers",
    "Compiler Infrastructure",
    "Release Engineering",
    "Graphics Drivers",
    "Embedded Systems",
    "Cloud Infrastructure",
    "Payments Platform",
    "Crypto Libraries",
    "Firmware Validation",
    "Identity Services",
    "Search Relevance",
    "Mobile Foundations",
];

const US_CITIES: &[&str] = &[
    "San Antonio",
    "Austin",
    "Denver",
    "Portland",
    "Columbus",
    "Sacramento",
    "Raleigh",
    "Tucson",
    "Boise",
    "Madison",
    "Albany",
    "Richmond",
    "Omaha",
    "Spokane",
    "Savannah",
];

const US_STATES: &[&str] = &[
    "Texas",
    "Ohio",
    "Oregon",
    "Colorado",
    "Virginia",
    "Michigan",
    "Arizona",
    "Maryland",
    "Minnesota",
    "Wisconsin",
];

const GB_PLACES: &[&str] = &[
    "Leeds",
    "Bristol",
    "Nottingham",
    "Sheffield",
    "Cardiff",
    "Aberdeen",
    "Plymouth",
    "Norwich",
    "Swansea",
    "Southampton",
    "Kent",
    "Devon",
    "Lancashire",
    "Norfolk",
    "Cornwall",
];

const CAPITALS: &[&str] = &[
    "Paris", "Madrid", "Tokyo", "Tbilisi", "Berlin", "Ottawa", "Nairobi", "Lima", "Hanoi", "Oslo",
];

const DE_CITIES: &[&str] = &[
    "München",
    "Hamburg",
    "Köln",
    "Stuttgart",
    "Düsseldorf",
    "Leipzig",
    "Dresden",
    "Hannover",
    "Nürnberg",
    "Bremen",
    "Augsburg",
    "Freiburg",
    "Mainz",
    "Kassel",
    "Rostock",
    "Würzburg",
];

/// German topics, written after `für`.
const DE_TOPICS: &[&str] = &[
    "Verbraucherschutz",
    "Stadtentwicklung",
    "Umwelt",
    "Digitalisierung",
    "Energie",
    "Bildung",
    "Gesundheit",
    "Kultur",
    "Wohnen",
    "Integration",
    "Verkehr",
    "Wirtschaft",
];

const JP_CITIES: &[&str] = &[
    "横浜",
    "大阪",
    "名古屋",
    "札幌",
    "神戸",
    "京都",
    "福岡",
    "川崎",
    "仙台",
    "広島",
    "千葉",
    "静岡",
    "熊本",
    "岡山",
    "新潟",
    "浜松",
    "金沢",
];

const JP_PREFECTURES: &[&str] = &[
    "神奈川",
    "埼玉",
    "千葉",
    "愛知",
    "兵庫",
    "福岡",
    "静岡",
    "広島",
    "宮城",
    "新潟",
];

/// Japanese topics of named sections (`課`) and bureaus (`局`).
const JP_TOPICS: &[&str] = &[
    "情報政策",
    "産業振興",
    "環境保全",
    "都市計画",
    "危機管理",
    "国際交流",
    "資産税",
    "道路整備",
    "子育て支援",
    "文化振興",
];

/// Georgian units of public bodies.
const GE_UNITS: &[&str] = &[
    "საბაჟო დეპარტამენტი",
    "საგადასახადო დავების დეპარტამენტი",
    "სტატისტიკის დეპარტამენტი",
    "საერთაშორისო ურთიერთობების დეპარტამენტი",
    "Customs Department",
    "Tax Disputes Department",
];

/// Named EU units as directory pages write them: `Unit C.2 – Road Safety`.
fn eu_unit(rng: &mut ChaCha8Rng, topic: &str) -> String {
    let letter = *[b'A', b'B', b'C', b'D', b'E'].choose(rng).unwrap_or(&b'A') as char;
    let dash = *["–", "-"].choose(rng).unwrap_or(&"–");
    format!("Unit {letter}.{} {dash} {topic}", rng.random_range(1..=6))
}

/// A body's acronym from the capitals of its words, skipping function words: `Office of Road
/// Safety` → `ORS`. `None` when that gives fewer than two letters.
pub fn initials(name: &str) -> Option<String> {
    let out: String = name
        .split([' ', '-'])
        .filter_map(|w| w.chars().next())
        .filter(|c| c.is_ascii_uppercase())
        .collect();
    (out.len() >= 2).then_some(out)
}

/// Whether `name` belongs to `split`: 80% train, 10% valid, 10% test, stable across runs.
fn in_split(name: &str, split: Split) -> bool {
    let bucket = fnv1a(name.as_bytes(), 0x626f_6469) % 10;
    match split {
        Split::Train => bucket < 8,
        Split::Valid => bucket == 8,
        Split::Test => bucket == 9,
    }
}

fn split_pick<'t>(items: &'t [&'t str], split: Split, rng: &mut ChaCha8Rng) -> &'t str {
    let own: Vec<&&str> = items.iter().filter(|s| in_split(s, split)).collect();
    own.choose(rng)
        .map(|s| **s)
        .or_else(|| items.choose(rng).copied())
        .unwrap_or("")
}

fn listed(country: &str, split: Split) -> Vec<Body> {
    LISTED
        .iter()
        .filter(|(c, _)| *c == country)
        .flat_map(|(_, bodies)| bodies.iter())
        .filter(|(name, _)| in_split(name, split))
        .map(|(name, acronym)| Body {
            name: name.to_string(),
            acronym: (!acronym.is_empty()).then(|| acronym.to_string()),
        })
        .collect()
}

/// The public bodies of one (split, country): its listed bodies and the EU's.
#[derive(Debug, Default, Clone)]
pub struct Bodies {
    country: &'static str,
    split: Option<Split>,
    own: Vec<Body>,
    eu: Vec<Body>,
}

impl Bodies {
    pub fn new(country: &'static str, split: Split) -> Bodies {
        Bodies {
            country,
            split: Some(split),
            own: listed(country, split),
            eu: listed("EU", split),
        }
    }

    fn split(&self) -> Split {
        self.split.unwrap_or(Split::Train)
    }

    /// A public body: a listed one half the time (an EU one for about one in ten), otherwise
    /// one composed from a pattern. With `acronym`, only a body that has one.
    pub fn body(&self, acronym: bool, rng: &mut ChaCha8Rng) -> Body {
        for _ in 0..64 {
            let b = if rng.random_bool(0.5) {
                let list = if rng.random_bool(0.2) || self.own.is_empty() {
                    &self.eu
                } else {
                    &self.own
                };
                match list.choose(rng) {
                    Some(b) => b.clone(),
                    None => self.composed(rng),
                }
            } else {
                self.composed(rng)
            };
            if !acronym || b.acronym.is_some() {
                return b;
            }
        }
        // A split whose lists hold no acronym at all still gets one, composed.
        let name = format!("Office of {}", split_pick(TOPICS, self.split(), rng));
        Body {
            acronym: initials(&name),
            name,
        }
    }

    fn composed(&self, rng: &mut ChaCha8Rng) -> Body {
        let split = self.split();
        let topic = split_pick(TOPICS, split, rng);
        let plain = |name: String| Body {
            name,
            acronym: None,
        };
        let abbreviated = |name: String| {
            let acronym = initials(&name);
            Body { name, acronym }
        };
        match self.country {
            "US" => match rng.random_range(0..6) {
                0 => abbreviated(format!("Office of {topic}")),
                1 => abbreviated(format!("Bureau of {topic}")),
                2 => plain(format!("City of {}", split_pick(US_CITIES, split, rng))),
                3 => plain(format!(
                    "{} Department of {topic}",
                    split_pick(US_STATES, split, rng)
                )),
                4 => abbreviated(format!("National {topic} Center")),
                _ => abbreviated(format!("Center for {topic}")),
            },
            "GB" => match rng.random_range(0..6) {
                0 => plain(format!(
                    "{} City Council",
                    split_pick(GB_PLACES, split, rng)
                )),
                1 => plain(format!(
                    "{} County Council",
                    split_pick(GB_PLACES, split, rng)
                )),
                2 => plain(format!(
                    "British Embassy {}",
                    split_pick(CAPITALS, split, rng)
                )),
                3 => plain(format!("{} Crown Court", split_pick(GB_PLACES, split, rng))),
                4 => abbreviated(format!("Office for {topic}")),
                _ => plain(format!(
                    "{} Marine Office",
                    split_pick(GB_PLACES, split, rng)
                )),
            },
            "DE" => {
                let city = split_pick(DE_CITIES, split, rng);
                match rng.random_range(0..7) {
                    0 => plain(format!("Amtsgericht {city}")),
                    1 => plain(format!("Landgericht {city}")),
                    2 => plain(format!("Finanzamt {city}")),
                    3 => plain(format!("Stadt {city}")),
                    4 => plain(format!("Landratsamt {city}")),
                    5 => Body {
                        name: format!("Industrie- und Handelskammer {city}"),
                        acronym: Some(format!("IHK {city}")),
                    },
                    _ => plain(format!(
                        "Senatsverwaltung für {}",
                        split_pick(DE_TOPICS, split, rng)
                    )),
                }
            }
            "JP" => {
                let city = split_pick(JP_CITIES, split, rng);
                plain(match rng.random_range(0..6) {
                    0 => format!("{city}市役所"),
                    1 => format!("{city}市"),
                    2 => format!("{}県庁", split_pick(JP_PREFECTURES, split, rng)),
                    3 => format!("{city}地方裁判所"),
                    4 => format!("{city}税務署"),
                    _ => format!("{city}市{}局", split_pick(JP_TOPICS, split, rng)),
                })
            }
            _ => match self.own.choose(rng) {
                Some(b) => b.clone(),
                None => abbreviated(format!("Office of {topic}")),
            },
        }
    }

    /// The body a company is registered with, as imprints and footers name it: a register
    /// court, Companies House, a state's corporations division, a legal affairs bureau.
    pub fn registry(&self, rng: &mut ChaCha8Rng) -> String {
        let split = self.split();
        match self.country {
            "DE" => format!("Amtsgericht {}", split_pick(DE_CITIES, split, rng)),
            "GB" => "Companies House".to_string(),
            "US" => format!(
                "{} Division of Corporations",
                split_pick(US_STATES, split, rng)
            ),
            "JP" => format!("{}地方法務局", split_pick(JP_CITIES, split, rng)),
            "GE" if rng.random_bool(0.5) => "საჯარო რეესტრის ეროვნული სააგენტო".to_string(),
            _ => "National Agency of Public Registry".to_string(),
        }
    }

    /// A named sub-unit: an office, division, directorate, section, or team with its own name.
    /// Generic business functions (`Customer Service`, `経理部`) are not units; they stay
    /// negatives in `templates::NEG_DEPARTMENTS`.
    pub fn unit(&self, rng: &mut ChaCha8Rng) -> String {
        let split = self.split();
        let topic = split_pick(TOPICS, split, rng);
        match self.country {
            "DE" if rng.random_bool(0.6) => {
                let topic = split_pick(DE_TOPICS, split, rng);
                match rng.random_range(0..3) {
                    0 => format!("Referat {topic}"),
                    1 => format!("Abteilung {topic}"),
                    _ => format!("Stabsstelle {topic}"),
                }
            }
            "JP" if rng.random_bool(0.7) => match rng.random_range(0..3) {
                0 => format!(
                    "第{}{}部",
                    rng.random_range(1..=5),
                    ["技術", "営業", "開発", "製造"]
                        .choose(rng)
                        .copied()
                        .unwrap_or("技術")
                ),
                1 => format!("{}課", split_pick(JP_TOPICS, split, rng)),
                _ => format!("{}室", split_pick(JP_TOPICS, split, rng)),
            },
            "GE" if rng.random_bool(0.6) => split_pick(GE_UNITS, split, rng).to_string(),
            _ => match rng.random_range(0..8) {
                0 => format!("Office of {topic}"),
                1 => format!("Division of {topic}"),
                2 => format!("{topic} Division"),
                3 => format!("{topic} Branch"),
                4 => format!("{topic} Directorate"),
                5 => eu_unit(rng, topic),
                _ => format!("{} Team", split_pick(TEAM_TOPICS, split, rng)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn initials_skip_function_words() {
        assert_eq!(initials("Office of Road Safety").as_deref(), Some("ORS"));
        assert_eq!(
            initials("Center for Research and Innovation").as_deref(),
            Some("CRI")
        );
        assert_eq!(initials("Enforcement"), None);
    }

    #[test]
    fn acronym_draws_always_have_one_and_every_split_has_bodies() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for country in ["US", "GB", "DE", "GE", "JP"] {
            for split in [Split::Train, Split::Valid, Split::Test] {
                let b = Bodies::new(country, split);
                for _ in 0..50 {
                    assert!(b.body(true, &mut rng).acronym.is_some(), "{country}");
                    assert!(!b.body(false, &mut rng).name.is_empty());
                    assert!(!b.unit(&mut rng).is_empty());
                }
            }
        }
    }

    #[test]
    fn listed_bodies_fall_in_exactly_one_split() {
        for (_, bodies) in LISTED {
            for (name, _) in *bodies {
                let n = [Split::Train, Split::Valid, Split::Test]
                    .into_iter()
                    .filter(|&s| in_split(name, s))
                    .count();
                assert_eq!(n, 1, "{name}");
            }
        }
    }
}
