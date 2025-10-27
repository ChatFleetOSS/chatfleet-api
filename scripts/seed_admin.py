import asyncio
import os
from datetime import datetime, timezone
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from app.services.users import hash_password

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/chatfleet")

async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client.get_default_database()
    col = db["users"]
    email = "admin@chatfleet.local"
    existing = await col.find_one({"email": email})
    now = datetime.now(timezone.utc)
    doc = {
        "email": email,
        "password_hash": hash_password("adminpass"),
        "name": "Seed Admin",
        "role": "admin",
        "rags": [],
        "created_at": now,
        "updated_at": now,
    }
    if existing:
        await col.replace_one({"_id": existing["_id"]}, doc)
        print("Updated existing admin", existing["_id"])
    else:
        result = await col.insert_one(doc)
        print("Inserted admin", result.inserted_id)

asyncio.run(main())
