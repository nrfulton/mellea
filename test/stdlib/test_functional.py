import pytest

from mellea.backends import ModelOption
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Message
from mellea.stdlib.functional import aact, ainstruct, avalidate, instruct
from mellea.stdlib.requirements import req
from mellea.stdlib.session import start_session
from mellea import MelleaSession
import mellea.stdlib.functional as mfuncs
from mellea.core import CBlock, ModelOutputThunk


@pytest.fixture(scope="module")
def m_session(gh_run):
    m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 5})
    yield m
    del m


def test_func_context(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    out, ctx = instruct("Write a sentence.", initial_ctx, backend)
    assert initial_ctx is not ctx
    assert ctx._data is out


async def test_aact(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    out, ctx = await aact(Message(role="user", content="hello"), initial_ctx, backend)

    assert initial_ctx is not ctx
    assert ctx._data is out


async def test_ainstruct(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    out, ctx = await ainstruct("Write a sentence", initial_ctx, backend)

    assert initial_ctx is not ctx
    assert ctx._data is out


async def test_avalidate(m_session):
    initial_ctx = m_session.ctx
    backend = m_session.backend

    val_result = await avalidate(
        reqs=[req("Be formal."), req("Avoid telling jokes.")],
        context=initial_ctx,
        backend=backend,
        output=ModelOutputThunk("Here is an output."),
    )

    assert len(val_result) == 2
    assert val_result[0] is not None


@pytest.mark.qualitative
async def test_aact_on_cblock(m_session):
    m: MelleaSession = m_session
    backend, ctx = m.backend, m.ctx  # type: ignore
    result, _ = mfuncs.act(CBlock("What is 1+1?"), ctx, backend)
    assert "2" in result.value or "two" in result.value


@pytest.mark.qualitative
async def test_aact_on_mot(m_session):
    m: MelleaSession = m_session
    backend, ctx = m.backend, m.ctx  # type: ignore
    mot = ModelOutputThunk(value="1+1=2")
    result, _ = mfuncs.act(mot, ctx, backend)
    assert "2" in result.value or "two" in result.value


if __name__ == "__main__":
    pytest.main([__file__])
