//! US organization names and units used by the synthetic document generator.
//! Public body names remain held until the US roster and evaluation overlap are checked.

use std::collections::HashSet;

use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;
use tessera::internal::fnv1a;

use crate::data::Split;

/// A public body and its acronym, when a reviewed roster becomes available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Body {
    pub name: String,
    pub acronym: Option<String>,
}

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

const US_CHARITIES: &[&str] = &[
    "United Way",
    "Habitat for Humanity",
    "American Red Cross",
    "Feeding America",
    "Goodwill Industries",
    "The Salvation Army",
    "YMCA of the USA",
    "Boys & Girls Clubs of America",
    "Big Brothers Big Sisters",
    "March of Dimes",
    "Special Olympics",
    "Meals on Wheels America",
    "St. Jude Children's Research Hospital",
    "Direct Relief",
    "Doctors Without Borders USA",
];

const US_MEDIA: &[&str] = &[
    "CNN",
    "Fox News",
    "NPR",
    "The New York Times",
    "The Associated Press",
    "Politico",
    "The Washington Post",
    "Bloomberg",
    "CBS News",
    "ABC News",
];

const US_UNIVERSITIES: &[&str] = &[
    "Ohio State University",
    "University of Texas at Austin",
    "Georgia Institute of Technology",
    "University of Michigan",
    "University of California, Berkeley",
    "Arizona State University",
];

const US_PARTIES: &[&str] = &["Democratic Party", "Republican Party"];

fn in_split(name: &str, split: Split) -> bool {
    let bucket = fnv1a(name.as_bytes(), 0x626f_6469) % 10;
    match split {
        Split::Train => bucket < 8,
        Split::Valid => bucket == 8,
        Split::Test => bucket == 9,
    }
}

fn split_pick<'a>(items: &'a [&'a str], split: Split, rng: &mut ChaCha8Rng) -> &'a str {
    let own: Vec<_> = items
        .iter()
        .copied()
        .filter(|name| in_split(name, split))
        .collect();
    own.choose(rng).copied().unwrap_or(items[0])
}

fn pick(items: &[&str], rng: &mut ChaCha8Rng) -> String {
    items.choose(rng).copied().unwrap_or("").to_string()
}

/// Split-owned foreign public-body names are absent from the US generator.
pub(crate) fn reserved_listed_surfaces() -> HashSet<&'static str> {
    HashSet::new()
}

/// US organization vocabulary for one synthetic data split.
#[derive(Debug, Clone)]
pub struct Bodies {
    split: Split,
}

impl Default for Bodies {
    fn default() -> Self {
        Self {
            split: Split::Train,
        }
    }
}

impl Bodies {
    /// Creates the US vocabulary for one split.
    pub fn new(country: &'static str, split: Split) -> Bodies {
        assert_eq!(country, "US", "only US synthetic bodies are supported");
        Bodies { split }
    }

    /// Public body names require a reviewed roster before use in synthetic training data.
    pub fn body(&self, _acronym: bool, _rng: &mut ChaCha8Rng) -> anyhow::Result<Body> {
        anyhow::bail!("no approved US public-body roster")
    }

    /// Georgian local office headings are outside the US generator.
    pub fn local_office(&self, _rng: &mut ChaCha8Rng) -> Option<String> {
        None
    }

    /// Georgian street fragments are outside the US generator.
    pub fn incomplete_street(&self, _rng: &mut ChaCha8Rng) -> Option<String> {
        None
    }

    /// A US state corporation registry.
    pub fn registry(&self, rng: &mut ChaCha8Rng) -> anyhow::Result<String> {
        Ok(format!(
            "{} Division of Corporations",
            split_pick(US_STATES, self.split, rng)
        ))
    }

    /// A named unit within an organization.
    pub fn chain(&self, rng: &mut ChaCha8Rng) -> anyhow::Result<String> {
        self.unit(rng)
    }

    /// A US charity name.
    pub fn charity(&self, rng: &mut ChaCha8Rng) -> String {
        pick(US_CHARITIES, rng)
    }

    /// A US university name.
    pub fn university(&self, rng: &mut ChaCha8Rng) -> String {
        pick(US_UNIVERSITIES, rng)
    }

    /// A committee or council name assembled from split-owned topics.
    pub fn council(&self, rng: &mut ChaCha8Rng) -> String {
        let topic = split_pick(TOPICS, self.split, rng);
        let suffix = ["Committee", "Advisory Board", "Task Force", "Commission"]
            .choose(rng)
            .copied()
            .unwrap_or("Committee");
        format!("{topic} {suffix}")
    }

    /// A US political party name.
    pub fn party(&self, rng: &mut ChaCha8Rng) -> String {
        pick(US_PARTIES, rng)
    }

    /// A US news organization name.
    pub fn media(&self, rng: &mut ChaCha8Rng) -> String {
        pick(US_MEDIA, rng)
    }

    /// A named office, division, or team.
    pub fn unit(&self, rng: &mut ChaCha8Rng) -> anyhow::Result<String> {
        let topic = split_pick(TOPICS, self.split, rng);
        Ok(match rng.random_range(0..7) {
            0 => format!("Office of {topic}"),
            1 => format!("Division of {topic}"),
            2 => format!("{topic} Division"),
            3 => format!("{topic} Branch"),
            4 => format!("{topic} Directorate"),
            _ => format!("{} Team", split_pick(TEAM_TOPICS, self.split, rng)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn public_body_draws_stay_held() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for split in Split::ALL {
            let bodies = Bodies::new("US", split);
            assert!(bodies.body(false, &mut rng).is_err());
            assert!(bodies.body(true, &mut rng).is_err());
        }
    }

    #[test]
    fn us_vocabulary_does_not_draw_foreign_parties_or_universities() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let bodies = Bodies::new("US", Split::Train);
        for _ in 0..200 {
            assert!(US_PARTIES.contains(&bodies.party(&mut rng).as_str()));
            assert!(US_UNIVERSITIES.contains(&bodies.university(&mut rng).as_str()));
        }
    }
}
