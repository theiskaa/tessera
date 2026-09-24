//! Word forms a template slot can ask for: Georgian case endings, which are glued to the last
//! word of a name and stay inside its span (`თამარ იოსელიანმა`, `შემოსავლების სამსახურის`),
//! and the English possessive, which follows the span (`HMRC` then `’s`).

use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// The form of a slot's value: `{person:erg#1}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Plain,
    /// Georgian ergative (`-მა`, `-მ`): the subject of a past transitive verb.
    Erg,
    /// Georgian genitive (`-ის`, `-ს`).
    Gen,
    /// Georgian dative (`-ს`).
    Dat,
    /// Georgian dative with the postposition `-თან` ("with, at").
    Com,
    /// Georgian `-ში` ("in").
    Loc,
    /// English possessive `'s` or `’s`.
    Poss,
}

impl Form {
    /// The form a slot names after its colon: `erg`, `gen`, `dat`, `com`, `loc`, or `poss`.
    pub fn parse(name: &str) -> Option<Form> {
        Some(match name {
            "erg" => Form::Erg,
            "gen" => Form::Gen,
            "dat" => Form::Dat,
            "com" => Form::Com,
            "loc" => Form::Loc,
            "poss" => Form::Poss,
            _ => return None,
        })
    }
}

/// Whether `c` is a Georgian Mkhedruli letter.
pub(crate) fn georgian(c: char) -> bool {
    ('\u{10D0}'..='\u{10FF}').contains(&c)
}

const QUOTES: [char; 4] = ['"', '“', '”', '»'];

/// `value` in `form`, and any part of the form that belongs outside the span.
///
/// Georgian endings go on the last word: glued to a Georgian word by its final letter, after
/// a hyphen on a Latin word or a short acronym (`Air China-ს`, `თსუ-ს`), and inside the
/// closing quote of a quoted name. In an organization's name an adjective before the head
/// noun declines with it (`საქართველოს ეროვნულმა ბანკმა`); `person` tells a proper name,
/// whose genitive keeps its final `-ა` (`ლუკას`), from a common noun (`მერია` → `მერიის`).
///
/// The English possessive, a token of its own, always follows the span.
pub fn apply(
    value: &str,
    form: Form,
    person: bool,
    rng: &mut ChaCha8Rng,
) -> (String, Option<&'static str>) {
    match form {
        Form::Plain => (value.to_string(), None),
        Form::Poss => {
            let suffix = if rng.random_bool(0.5) { "’s" } else { "'s" };
            (value.to_string(), Some(suffix))
        }
        _ => (georgian_form(value, form, person), None),
    }
}

fn georgian_form(value: &str, form: Form, person: bool) -> String {
    if let Some(inner) = value.strip_suffix(QUOTES)
        && let Some(close) = value.chars().last()
    {
        return format!("{}{close}", georgian_form(inner, form, person));
    }
    let mut words: Vec<String> = value.split(' ').map(str::to_string).collect();
    let Some(last) = words.pop() else {
        return value.to_string();
    };
    if !person {
        // Only the adjectives right before the head noun agree with it; an adjective before a
        // genitive (`ფინანსური მონიტორინგის სამსახური`) belongs to that genitive and stays.
        for word in words.iter_mut().rev() {
            if !is_adjective(word) {
                break;
            }
            *word = adjective(word, form);
        }
    }
    words.push(inflect_word(&last, form, person));
    words.join(" ")
}

/// Whether `word` has an adjective ending that declines with its noun: `ეროვნული`,
/// `ცენტრალური`, `უმაღლესი`, `სახელმწიფოებრივი`.
fn is_adjective(word: &str) -> bool {
    [
        "ული",
        "ური",
        "ელი",
        "ადი",
        "ივი",
        "ალი",
        "არი",
        "ესი",
        "ებრივი",
    ]
    .iter()
    .any(|end| word.ends_with(end) && word.chars().count() > end.chars().count() + 1)
}

/// An adjective agreeing with its noun: `-მა` in the ergative, its `-ი` dropped in the dative
/// and before `-თან` and `-ში`, unchanged in the genitive.
fn adjective(word: &str, form: Form) -> String {
    let stem = &word[..word.len() - 'ი'.len_utf8()];
    match form {
        Form::Erg => format!("{stem}მა"),
        Form::Dat | Form::Com | Form::Loc => stem.to_string(),
        Form::Gen | Form::Plain | Form::Poss => word.to_string(),
    }
}

fn inflect_word(word: &str, form: Form, person: bool) -> String {
    let Some(end) = word.chars().last() else {
        return word.to_string();
    };
    let acronym = word.chars().count() <= 3 && word.chars().all(georgian);
    if !georgian(end) || acronym {
        let suffix = match form {
            Form::Erg => "მა",
            Form::Gen => "ის",
            Form::Dat => "ს",
            Form::Com => "თან",
            Form::Loc => "ში",
            Form::Plain | Form::Poss => return word.to_string(),
        };
        return format!("{word}-{suffix}");
    }
    let stem = &word[..word.len() - end.len_utf8()];
    match end {
        // Consonant stems carry `-ი` in the nominative, dropped before every ending.
        'ი' => match form {
            Form::Erg => format!("{stem}მა"),
            Form::Gen => format!("{stem}ის"),
            Form::Dat => format!("{stem}ს"),
            Form::Com => format!("{stem}თან"),
            Form::Loc => format!("{stem}ში"),
            Form::Plain | Form::Poss => word.to_string(),
        },
        'ე' | 'ა' => match form {
            Form::Erg => format!("{word}მ"),
            Form::Gen if person && end == 'ა' => format!("{word}ს"),
            Form::Gen => format!("{stem}ის"),
            Form::Dat => format!("{word}ს"),
            Form::Com => format!("{word}სთან"),
            Form::Loc => format!("{word}ში"),
            Form::Plain | Form::Poss => word.to_string(),
        },
        'ო' | 'უ' => match form {
            Form::Erg => format!("{word}მ"),
            Form::Gen | Form::Dat => format!("{word}ს"),
            Form::Com => format!("{word}სთან"),
            Form::Loc => format!("{word}ში"),
            Form::Plain | Form::Poss => word.to_string(),
        },
        // A consonant at the end (`თამარ`): the endings of a consonant stem.
        _ => match form {
            Form::Erg => format!("{word}მა"),
            Form::Gen => format!("{word}ის"),
            Form::Dat => format!("{word}ს"),
            Form::Com => format!("{word}თან"),
            Form::Loc => format!("{word}ში"),
            Form::Plain | Form::Poss => word.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn georgian_endings_follow_the_last_letter() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut f = |v: &str, form, person| apply(v, form, person, &mut rng).0;
        assert_eq!(f("თამარ იოსელიანი", Form::Erg, true), "თამარ იოსელიანმა");
        assert_eq!(f("ირაკლი კობახიძე", Form::Com, true), "ირაკლი კობახიძესთან");
        assert_eq!(f("ირაკლი კობახიძე", Form::Gen, true), "ირაკლი კობახიძის");
        assert_eq!(
            f("შემოსავლების სამსახური", Form::Gen, false),
            "შემოსავლების სამსახურის"
        );
        assert_eq!(f("თბილისის მერია", Form::Gen, false), "თბილისის მერიის");
        assert_eq!(
            f("ფინანსთა სამინისტრო", Form::Loc, false),
            "ფინანსთა სამინისტროში"
        );
        assert_eq!(f("ლუკა", Form::Gen, true), "ლუკას");
        assert_eq!(f("Air China", Form::Dat, false), "Air China-ს");
        assert_eq!(f("Acme", Form::Plain, false), "Acme");
        assert_eq!(
            f("საჯარო რეესტრის ეროვნული სააგენტო", Form::Com, false),
            "საჯარო რეესტრის ეროვნულ სააგენტოსთან"
        );
        assert_eq!(
            f("საქართველოს ეროვნული ბანკი", Form::Erg, false),
            "საქართველოს ეროვნულმა ბანკმა"
        );
        assert_eq!(
            f("მთავარი ბუღალტერი", Form::Erg, false),
            "მთავარმა ბუღალტერმა"
        );
        assert_eq!(f("თსუ", Form::Dat, false), "თსუ-ს");
        assert_eq!(
            f("ფინანსური მონიტორინგის სამსახური", Form::Erg, false),
            "ფინანსური მონიტორინგის სამსახურმა"
        );
        assert_eq!(f("თიბისი ბანკი", Form::Erg, false), "თიბისი ბანკმა");
        assert_eq!(
            f("სს \"კლირინგ ცენტრი\"", Form::Gen, false),
            "სს \"კლირინგ ცენტრის\""
        );
    }

    #[test]
    fn possessives_take_an_apostrophe() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..20 {
            let (p, outside) = apply("HMRC", Form::Poss, false, &mut rng);
            assert_eq!(p, "HMRC");
            assert!(outside == Some("'s") || outside == Some("’s"));
        }
        let (p, outside) = apply("ანთაძე", Form::Poss, true, &mut rng);
        assert_eq!(p, "ანთაძე");
        assert!(outside.is_some());
    }
}
