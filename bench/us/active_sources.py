"""Real silver files used by the current US detector configuration."""


BASE_SILVER = (
    ("us-reviewed-strict-v1", "evaluation_gold_sha256"),
    ("us-house-staff-v1", "evaluation_sha256"),
    ("us-house-district-offices-v1", "evaluation_sha256"),
    ("us-reviewed-contact-snippets-v1", "evaluation_gold_sha256"),
    ("us-reviewed-org-paragraphs-v1", "evaluation_gold_sha256"),
    ("us-r23-reviewed-org-v1", "evaluation_gold_sha256"),
    ("us-federal-org-reviewed-v2", "evaluation_gold_sha256"),
    ("us-park-contacts-v1", "evaluation_gold_sha256"),
    ("us-or-districts-v1", "evaluation_gold_sha256"),
    ("us-2025-year-contacts-v2", "evaluation_sha256"),
    ("us-2025-year-acronym-prose-v3", "evaluation_sha256"),
    ("us-2025-year-contact-expansion-v1", "evaluation_sha256"),
)

SAFE_REPLACEMENTS = {
    "us-reviewed-strict-v1": "us-reviewed-strict-safe-v1",
    "us-reviewed-contact-snippets-v1": "us-reviewed-contact-snippets-safe-v1",
    "us-reviewed-org-paragraphs-v1": "us-reviewed-org-paragraphs-safe-v1",
    "us-r23-reviewed-org-v1": "us-r23-reviewed-org-safe-v1",
    "us-federal-org-reviewed-v2": "us-federal-org-reviewed-safe-v1",
    "us-2025-year-contact-expansion-v1": "us-2025-year-contact-expansion-safe-v1",
}

ACTIVE_SILVER = tuple((SAFE_REPLACEMENTS.get(name, name), key)
                      for name, key in BASE_SILVER) + (
    ("us-2025-targeted-contacts-v1", "evaluation_sha256"),
)
