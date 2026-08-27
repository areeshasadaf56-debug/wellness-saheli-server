"""
auth.py

Implements the three account endpoints the Flutter app already calls
(see lib/providers/cycle_provider.dart: signUp / signIn / resetPassword)
but that had no backend implementation at all.

Design notes for your viva:
  - Passwords are never stored in plain text. They're hashed with
    bcrypt (via passlib) -- a slow, salted hash designed specifically
    for passwords, unlike a fast general-purpose hash like SHA-256.
  - There is deliberately NO session token / JWT returned here. Look
    at cycle_provider.dart: after signUp/signIn succeed, the app just
    calls login(name) and stores a local "isLoggedIn" flag +
    display name in SharedPreferences. It never attaches an
    Authorization header to any later request (check
    health_profile_service.dart, pcos_api_service.dart, etc. -- none
    of them send a token). So this account system authenticates
    ONCE at sign-in time but does not gate any other endpoint.
    That's a real limitation (see SECURITY NOTES below) -- adding JWT
    issuance here without the app ever sending it back would be
    complexity with zero actual benefit, so it's intentionally left
    out rather than added for show.

SECURITY NOTES (be upfront about these in your report/viva -- an
examiner who reads main.py will notice immediately if you don't):
  1. /profile/{user_id} is NOT authenticated. Anyone who knows or
     guesses a user_id (a random UUID, so hard to *guess*, but not
     impossible to intercept) can read or overwrite that profile.
     This is acceptable for an FYP demo but must be listed as a
     known limitation, not silently ignored.
  2. There's no rate limiting on /signin, so this endpoint is
     vulnerable to online password-guessing if it were public and
     high-traffic. Fine for a low-traffic FYP demo; call it out as
     future work.
"""

import bcrypt

import database

# Using the `bcrypt` package directly rather than passlib's CryptContext
# wrapper. passlib 1.7.4 (its latest release) has a known incompatibility
# with bcrypt>=4.0 -- its internal self-test misreads bcrypt's version
# and raises a spurious "password cannot be longer than 72 bytes" error
# even for short passwords. Calling bcrypt directly sidesteps that
# entirely and is one fewer dependency.


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


class AuthError(Exception):
    """Raised for any expected auth failure; main.py turns this into
    the appropriate HTTP status code + `detail` message the Flutter
    app already expects (see cycle_provider.dart: `body['detail']`)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


def sign_up(name: str, email: str, password: str) -> str:
    """Creates a new account. Returns the display name on success."""
    if len(password) < 6:
        raise AuthError(422, "Password must be at least 6 characters.")

    if database.get_user_by_email(email) is not None:
        raise AuthError(409, "An account with this email already exists.")

    user = database.create_user(name.strip(), email, hash_password(password))
    return user["name"]


def sign_in(email: str, password: str) -> str:
    """Validates credentials. Returns the display name on success."""
    user = database.get_user_by_email(email)
    if user is None or not verify_password(password, user["password_hash"]):
        # Deliberately the SAME message for "no such email" and "wrong
        # password" -- distinguishing them lets an attacker enumerate
        # which emails have accounts.
        raise AuthError(401, "Incorrect email or password.")
    return user["name"]


def reset_password(email: str, new_password: str) -> None:
    if len(new_password) < 6:
        raise AuthError(422, "Password must be at least 6 characters.")

    user = database.get_user_by_email(email)
    if user is None:
        # NOTE: in a production system you would return this same
        # generic success response regardless of whether the email
        # exists, to avoid leaking which emails are registered. Kept
        # as an explicit error here only because there is no email-based
        # reset flow (no emailed reset link/token) -- the app currently
        # calls this endpoint directly with a new password, so silently
        # "succeeding" on a non-existent email would be confusing during
        # your own testing. Flag this trade-off if an examiner asks.
        raise AuthError(404, "No account found with this email.")

    database.update_password(email, hash_password(new_password))