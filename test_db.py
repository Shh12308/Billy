import asyncio
import asyncpg

DATABASE_URL = "postgresql://postgres:3kJy7ReRfyhdconb@db.orozxlbnurnchwodzfdt.supabase.co:5432/postgres"

async def test_connection():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected successfully!")
        await conn.close()
    except Exception as e:
        print("❌ Connection failed:")
        print(e)

asyncio.run(test_connection())
