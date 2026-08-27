"""
health_profile_api.py

Stores and retrieves each user's HealthProfile (the "diary" described in
health_profile.dart) as a JSON blob in SQLite, keyed by an anonymous
device id generated client-side (see health_profile_service.dart).

Deliberately schema-light: the profile is stored as a single JSON text
column rather than one column per field. This means adding new fields to
the Dart model (lifestyle, mental health flags, conversation log, etc.)
never requires a database migration here -- the backend just stores
whatever JSON it's given and hands it back unchanged. If you eventually
need to query/filter profiles server-side (e.g. "find all users flagged
high PCOS risk"), that's the point to introduce real columns for the
specific fields you need to query on -- not before.

Wire this into main_flask.py with:

    from health_profile_api import health_profile_bp
    app.register_blueprint(health_profile_bp)

Run once (or let init_db() below run automatically on import) to create
the table on first use -- no separate migration step needed for this
lightweight schema.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "app_deployment", "health_profiles.db"
)

health_profile_bp = Blueprint("health_profile", __name__)


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the health_profiles table if it doesn't exist yet."""
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS health_profiles (
            user_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# Run at import time so the table exists as soon as main.py starts,
# without needing a separate manual setup step (mirrors how
# pcos_app_model.joblib is loaded once at startup in main.py).
init_db()


@health_profile_bp.route("/profile/<user_id>", methods=["GET"])
def get_profile(user_id):
    """
    Returns the stored profile for this user_id, or a 404 if none
    exists yet (the Flutter service falls back to a blank local profile
    in that case -- see health_profile_service.dart's loadProfile()).
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT profile_json FROM health_profiles WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return jsonify({"detail": "No profile found for this user_id"}), 404

    return jsonify(json.loads(row["profile_json"]))


@health_profile_bp.route("/profile/<user_id>", methods=["PUT"])
def save_profile(user_id):
    """
    Upserts (creates or fully replaces) the profile for this user_id.
    The Flutter service always sends the FULL profile object on save
    (not a partial patch), so a straightforward replace is correct here
    -- partial-update logic lives client-side in
    HealthProfileService.updateProfile().
    """
    body = request.get_json(force=True) or {}

    # Basic sanity check -- don't silently accept a payload for a
    # different user_id than the URL says.
    if body.get("user_id") and body["user_id"] != user_id:
        return jsonify({
            "detail": f"user_id in body ('{body.get('user_id')}') does not match URL ('{user_id}')"
        }), 422

    body["user_id"] = user_id
    profile_json = json.dumps(body)
    updated_at = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO health_profiles (user_id, profile_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            profile_json = excluded.profile_json,
            updated_at = excluded.updated_at
        """,
        (user_id, profile_json, updated_at),
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "ok", "user_id": user_id, "updated_at": updated_at})


@health_profile_bp.route("/profile/<user_id>", methods=["DELETE"])
def delete_profile(user_id):
    """
    Wipes a user's stored profile. Useful for testing, and worth
    exposing in a real Settings screen later so users can clear their
    data -- especially given this profile can include self-reported
    stress/mental-health notes.
    """
    conn = _get_connection()
    conn.execute("DELETE FROM health_profiles WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted", "user_id": user_id})