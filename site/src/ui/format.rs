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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(462), "462");
        assert_eq!(thousands(1337), "1,337");
        assert_eq!(thousands(320115455), "320,115,455");
    }
}
