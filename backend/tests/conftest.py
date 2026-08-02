"""Shared test fixtures.

Everything runs against the real engine, a real SQLite database and real
filesystem storage. Nothing is mocked, because the previous suite mocked the
service layer entirely and therefore passed while the product was broken
end to end.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.analysis.engine import AnalysisEngine
from app.core.config import Environment, Settings, get_settings
from app.db import session as session_module
from app.domain.models import Transcript

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: a per-test database and blob root."""
    return Settings(
        environment=Environment.TEST,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        storage_local_path=tmp_path / "blobs",
        auth_enabled=False,
        log_level="CRITICAL",
        rate_limit_per_minute=0,
    )


@pytest.fixture
def engine() -> AnalysisEngine:
    """The lexicon engine — deterministic, no model downloads."""
    return AnalysisEngine()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A TestClient wired to isolated storage.

    The module-level engine singleton is reset around each test so a database
    from a previous test cannot leak into this one.
    """
    from app.main import create_app

    session_module._engine = None
    session_module._sessionmaker = None

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client

    session_module._engine = None
    session_module._sessionmaker = None
    get_settings.cache_clear()


# Keys for the auth-enabled client below. Must clear the 16-character minimum
# the registry enforces.
VIEWER_KEY = "cs_test_viewer_key_0000000000"
ANALYST_KEY = "cs_test_analyst_key_000000000"
ADMIN_KEY = "cs_test_admin_key_00000000000"


@pytest.fixture
def auth_client(settings: Settings) -> Iterator[TestClient]:
    """A TestClient with authentication actually switched on.

    The default ``client`` fixture runs with ``auth_enabled=False``, where every
    caller is an implicit admin -- so it can never catch an endpoint guarded by
    the wrong role. Nothing exercised the role wiring end to end before this.
    """
    from app.main import create_app

    session_module._engine = None
    session_module._sessionmaker = None

    secured = settings.model_copy(
        update={
            "auth_enabled": True,
            "api_keys": [
                f"{VIEWER_KEY}:viewer",
                f"{ANALYST_KEY}:analyst",
                f"{ADMIN_KEY}:admin",
            ],
        }
    )

    app = create_app(secured)
    with TestClient(app) as test_client:
        yield test_client

    session_module._engine = None
    session_module._sessionmaker = None
    get_settings.cache_clear()


@pytest.fixture
async def db_session(settings: Settings) -> AsyncIterator:
    """A bare database session for repository-level tests."""
    session_module._engine = None
    session_module._sessionmaker = None
    session_module.init_engine(settings.database_url)
    await session_module.create_schema()

    async with session_module.session_scope() as session:
        yield session

    await session_module.dispose_engine()


# -- Sample transcripts ---------------------------------------------------
#
# Written to exercise specific engine behaviour rather than to look realistic.
# Each one is named for the property it is meant to demonstrate.

# Fully qualified on every BANT dimension -- budget, authority, need and
# timeline are each stated explicitly by the prospect. This is the fixture
# that must score HOT; a call missing any one dimension should not.
QUALIFIED_CALL = """\
Rep: Hi Sarah, thanks for taking the time today.
Prospect: Sure, we have fifteen minutes.
Prospect: Honestly our current process is a mess. We're struggling with a \
completely manual workflow and it's costing us two days a week.
Rep: How long has that been the case?
Prospect: About a year. It's a top priority now because we go live in March.
Prospect: What's the pricing? We have budget allocated for this quarter.
Rep: Most teams your size land between forty and sixty thousand.
Prospect: That's within our budget. This is exactly what we need.
Prospect: I can approve anything under fifty thousand, so it's my decision.
Prospect: Can we set up a call next week to go through it?
Rep: Absolutely, I'll send an invite today.
Prospect: Great, looking forward to it.
"""

REJECTION_CALL = """\
Rep: Hi Mark, this is Priya calling about data tooling.
Prospect: I'm not interested. We already have something that works fine.
Prospect: We just signed with another vendor. Please take me off your list.
Rep: Understood, thanks for your time.
"""

OBJECTION_CALL = """\
Rep: Thanks for looking at the proposal.
Prospect: It's interesting but it's too expensive for us. That's out of our budget.
Rep: I understand. What is the current process costing you in engineering time?
Prospect: Maybe a day a week across two people.
Rep: So that's around fifty thousand a year in engineering time on something \
the platform automates for you, which is the comparison I would make.
Prospect: That's a fair point. But I need to check with my boss before anything.
"""

SHORT_CALL = "Rep: Hi there.\nProspect: Not now, thanks.\n"


@pytest.fixture
def qualified_transcript(engine: AnalysisEngine) -> Transcript:
    from app.analysis import parsing
    from app.analysis.diarization import infer_roles

    return infer_roles(parsing.parse_transcript(QUALIFIED_CALL))
