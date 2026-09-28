"""Email service - Outbound email triggers and OTP generation disabled.

Email verification is handled directly at intake using commercial verifier
techniques (syntax validation, typo detection, disposable email blocking,
dummy prefix rejection, and DNS MX/A mail server validation) without
triggering outbound emails.
"""

import logging

logger = logging.getLogger(__name__)

# Outbound email triggers and OTP generation have been removed per specification.
