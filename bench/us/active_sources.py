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

PRIOR_V3_SILVER = ACTIVE_SILVER + (
    ("us-r25-reviewed-contacts-v1", "evaluation_sha256"),
)

PRIOR_MILITARY_SILVER = PRIOR_V3_SILVER + (
    ("us-official-reviewed-contacts-v1", "evaluation_sha256"),
)

PRIOR_NRCS_SILVER = PRIOR_MILITARY_SILVER + (
    ("us-military-reviewed-addresses-v1", "evaluation_sha256"),
)

PRIOR_DOL_SILVER = PRIOR_NRCS_SILVER + (
    ("us-nrcs-reviewed-field-offices-v1", "evaluation_sha256"),
)

PRIOR_DOL_REMAINING_SILVER = PRIOR_DOL_SILVER + (
    ("us-dol-whd-reviewed-offices-v1", "evaluation_sha256"),
)

PRIOR_ENVIRONMENTAL_SILVER = PRIOR_DOL_REMAINING_SILVER + (
    ("us-dol-whd-reviewed-remaining-v1", "evaluation_sha256"),
)

PRIOR_MAINE_SILVER = PRIOR_ENVIRONMENTAL_SILVER + (
    ("us-environmental-hard-negatives-v1", "evaluation_sha256"),
)

V3_SILVER = PRIOR_MAINE_SILVER + (
    ("us-nrcs-maine-reviewed-offices-v1", "evaluation_sha256"),
)

V4_BASE_SILVER = tuple(
    ("us-reviewed-org-paragraphs-v4" if name == "us-reviewed-org-paragraphs-safe-v1"
     else name, key)
    for name, key in V3_SILVER
    if name not in {"us-park-contacts-v1", "us-or-districts-v1",
                    "us-nrcs-maine-reviewed-offices-v1"}
)

V4_SILVER = V4_BASE_SILVER + (
    ("us-v4-reviewed-additions-v2", "evaluation_sha256"),
    ("us-v4-reviewed-long-additions-v1", "evaluation_sha256"),
    ("us-v4-reviewed-historical-long-v1", "evaluation_sha256"),
    ("us-v4-reviewed-historical-acronyms-v1", "evaluation_sha256"),
)
