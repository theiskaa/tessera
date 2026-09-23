//! Number formatting for the page.

/// `n` with comma thousands separators, `1,337`.
pub(crate) fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A rate from 0 to 1 as a percentage with one decimal, `95.0%`.
pub(crate) fn percent(rate: f64) -> String {
    format!("{:.1}%", rate * 100.0)
}

/// Bytes in decimal units, `1.50 MB` or `95.1 KB`.
pub(crate) fn bytes(n: u64) -> String {
    let n = n as f64;
    if n >= 1e6 {
        format!("{:.2} MB", n / 1e6)
    } else {
        format!("{:.1} KB", n / 1e3)
    }
}

/// A count in words when it is small, as prose wants it.
pub(crate) fn words(n: u32) -> String {
    const SMALL: [&str; 11] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ];
    SMALL
        .get(n as usize)
        .map_or_else(|| n.to_string(), |w| (*w).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(462), "462");
        assert_eq!(thousands(1337), "1,337");
        assert_eq!(thousands(320115455), "320,115,455");
        assert_eq!(percent(0.9504), "95.0%");
        assert_eq!(bytes(1498542), "1.50 MB");
        assert_eq!(bytes(95093), "95.1 KB");
        assert_eq!(words(2), "two");
        assert_eq!(words(24), "24");
    }
}
