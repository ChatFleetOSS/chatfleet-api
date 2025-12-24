"""
Admin promotion intents: create and redeem at login/register.

This module lets the installer create a pending promotion for an email
address. When that user logs in (or registers), we atomically upgrade
their role to "admin" and mark the promotion as redeemed so that the
admin UI is available immediately.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId

from app.core.database import get_collection
from app.services.logging import write_system_log


def _col():
    return get_collection("admin_promotions")


async def ensure_indexes() -> None:
    col = _col()
    # Unique email constraint so we have at most one active intent per email.
    await col.create_index("email", unique=True)
    # TTL: expire at the timestamp in `expires_at` (Mongo treats 0 as use-date-field TTL)
    await col.create_index("expires_at", expireAfterSeconds=0)


async def create_promotion_intent(email: str, hours_valid: int = 48) -> None:
    """Upsert a promotion intent for the given email.

    This is primarily used by external tooling/installer. Idempotent.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=hours_valid)
    await _col().update_one(
        {"email": email.lower()},
        {
            "$setOnInsert": {
                "email": email.lower(),
                "created_at": now,
                "expires_at": expires,
                "redeemed": False,
            }
        },
        upsert=True,
    )


async def apply_if_pending(email: str, user_id: Optional[str] = None) -> bool:
    """If there is a pending promotion for this email, upgrade the user to admin.

    Returns True if a promotion was applied (or already redeemed for this user),
    False otherwise.
    """
    email_key = (email or "").lower().strip()
    if not email_key:
        return False

    promo = await _col().find_one({"email": email_key})
    if not promo:
        return False

    # Check expiry and redemption; TTL will clean up expired docs, but be explicit.
    now = datetime.now(timezone.utc)
    if promo.get("redeemed"):
        return False
    expires_at = promo.get("expires_at")
    # Normalize to aware UTC if Mongo returns naive datetimes
    if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at and expires_at < now:
        return False

    users = get_collection("users")
    user_doc = await users.find_one({"email": email_key})
    if not user_doc:
        # No user yet; keep the intent for later
        return False

    # Upgrade role if needed
    if user_doc.get("role") != "admin":
        await users.update_one({"_id": user_doc["_id"]}, {"$set": {"role": "admin", "updated_at": now}})

    # Mark promotion redeemed
    await _col().update_one(
        {"_id": promo["_id"]},
        {
            "$set": {
                "redeemed": True,
                "redeemed_at": now,
                "redeemed_by": ObjectId(user_id) if user_id else user_doc.get("_id"),
            }
        },
    )

    await write_system_log(
        event="auth.promotion.redeemed",
        user_id=str(user_id or user_doc.get("_id")),
        details={"email": email_key},
    )
    return True
