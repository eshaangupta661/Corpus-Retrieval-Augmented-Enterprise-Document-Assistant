from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

# The sync URI is postgresql://; the async engine needs the asyncpg driver.
_async_uri = str(settings.SQLALCHEMY_DATABASE_URI).replace(
    "postgresql://", "postgresql+asyncpg://", 1
)

async_engine = create_async_engine(_async_uri, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
        # Repositories only add/flush; commit once the request handler succeeds.
        await session.commit()
