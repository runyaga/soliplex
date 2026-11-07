import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import fastapi
import pytest
import yaml
from fastapi.testclient import TestClient

from soliplex import authz
from soliplex import installation
from soliplex import loggers
from soliplex import models
from soliplex.config import agents as config_agents
from soliplex.config import installation as config_installation
from soliplex.views import installation as installation_views

OLLAMA_BASE_URL = "http://ollama.example.com:11434"
TEST_MODEL_ONE = "test-model-one:1.2.3"
TEST_MODEL_TWO = "test-model-two:4.5.6"

GIT_HASH = "test-git-hash"
GIT_BRANCH = "test-git-branch"
GIT_TAG = "test-git-tag"

THE_USER_CLAIMS = {"email": "admin@example.com"}


@mock.patch("soliplex.views.installation.ConfigAuditLog")
def test_get_the_config_audit_log(cal_klass):
    found = installation_views.get_the_config_audit_log(THE_USER_CLAIMS)

    assert found is cal_klass.return_value
    cal_klass.assert_called_once_with(claims=THE_USER_CLAIMS)


@pytest.mark.anyio
@mock.patch(
    "soliplex.views.installation.models.InstallationInfo.from_config",
)
async def test_get_installation_info(ifc):
    i_config = mock.create_autospec(config_installation.InstallationConfig)
    the_installation = installation.Installation(i_config)
    the_logger = mock.create_autospec(loggers.LogWrapper)

    found = await installation_views.get_installation_info(
        the_installation=the_installation,
        the_logger=the_logger,
    )

    assert found is ifc.return_value

    ifc.assert_called_once_with(i_config)
    the_logger.debug.assert_called_once_with(
        loggers.INST_GET_INSTALLATION_INFO,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("w_admin_access", [False, True])
@mock.patch("soliplex.models.Installation.from_config")
async def test_get_installation(fc, w_admin_access):
    from soliplex import installation

    i_config = mock.create_autospec(config_installation.InstallationConfig)
    the_installation = installation.Installation(i_config)
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = w_admin_access
    the_logger = mock.create_autospec(loggers.LogWrapper)
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    if not w_admin_access:
        with pytest.raises(fastapi.HTTPException) as exc:
            await installation_views.get_installation(
                the_installation=the_installation,
                the_admin_users=the_admin_users,
                the_user_claims=THE_USER_CLAIMS,
                the_logger=the_logger,
                the_audit=the_audit,
            )

        assert exc.value.status_code == 403
        assert exc.value.detail == loggers.AUTHZ_ADMIN_ACCESS_REQUIRED
        the_audit.installation_read.assert_not_called()

    else:
        found = await installation_views.get_installation(
            the_installation=the_installation,
            the_admin_users=the_admin_users,
            the_user_claims=THE_USER_CLAIMS,
            the_logger=the_logger,
            the_audit=the_audit,
        )

        assert found is fc.return_value

        fc.assert_called_once_with(i_config)
        the_audit.installation_read.assert_called_once_with()

    the_admin_users.check_admin_access.assert_awaited_once_with(
        THE_USER_CLAIMS,
        resource=loggers.AUDIT_RESOURCE_INSTALLATION,
        action=loggers.AUDIT_ACTION_READ,
    )


@pytest.mark.anyio
async def test_reload_rooms_configs(
    monkeypatch,
):
    """An admin reload answers the added / removed / changed rooms."""
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)

    i_config = mock.create_autospec(config_installation.InstallationConfig)
    the_installation = installation.Installation(i_config)
    reloaded = installation.RoomConfigsReloaded(
        added=("new-room",),
        removed=("old-room",),
        changed=("changed-room",),
    )
    the_installation.reload_room_configs = mock.AsyncMock(
        return_value=reloaded,
    )
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = True
    the_logger = mock.create_autospec(loggers.LogWrapper)
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    found = await installation_views.reload_rooms_configs(
        the_installation=the_installation,
        the_admin_users=the_admin_users,
        the_user_claims=THE_USER_CLAIMS,
        the_logger=the_logger,
        the_audit=the_audit,
    )

    assert found == models.RoomConfigsReloaded(
        added=["new-room"],
        removed=["old-room"],
        changed=["changed-room"],
    )

    the_installation.reload_room_configs.assert_awaited_once_with()
    the_audit.rooms_reloaded.assert_called_once_with(
        added=["new-room"],
        removed=["old-room"],
        changed=["changed-room"],
    )

    the_admin_users.check_admin_access.assert_awaited_once_with(
        THE_USER_CLAIMS,
        resource=loggers.AUDIT_RESOURCE_INSTALLATION_ROOMS,
        action=loggers.AUDIT_ACTION_UPDATE,
    )

    bound_logger = the_logger.bind.return_value
    the_logger.bind.assert_called_once_with(loggers.AUTHZ_LOGGER_NAME)
    bound_logger.debug.assert_called_once_with(
        loggers.INST_POST_RELOAD_ROOMS,
    )


@pytest.mark.anyio
async def test_reload_rooms_configs_empty_delta(
    monkeypatch,
):
    """A reload which changes nothing answers empty lists."""
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)

    i_config = mock.create_autospec(config_installation.InstallationConfig)
    the_installation = installation.Installation(i_config)
    the_installation.reload_room_configs = mock.AsyncMock(
        return_value=installation.RoomConfigsReloaded(
            added=(),
            removed=(),
            changed=(),
        ),
    )
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = True
    the_logger = mock.create_autospec(loggers.LogWrapper)
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    found = await installation_views.reload_rooms_configs(
        the_installation=the_installation,
        the_admin_users=the_admin_users,
        the_user_claims=THE_USER_CLAIMS,
        the_logger=the_logger,
        the_audit=the_audit,
    )

    assert found == models.RoomConfigsReloaded()
    the_audit.rooms_reloaded.assert_called_once_with(
        added=[],
        removed=[],
        changed=[],
    )


@pytest.mark.anyio
async def test_reload_rooms_configs_not_admin(
    monkeypatch,
):
    """A non-admin reload answers 403 and reloads nothing."""
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)

    i_config = mock.create_autospec(config_installation.InstallationConfig)
    the_installation = installation.Installation(i_config)
    the_installation.reload_room_configs = mock.AsyncMock()
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = False
    the_logger = mock.create_autospec(loggers.LogWrapper)
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    with pytest.raises(fastapi.HTTPException) as exc:
        await installation_views.reload_rooms_configs(
            the_installation=the_installation,
            the_admin_users=the_admin_users,
            the_user_claims=THE_USER_CLAIMS,
            the_logger=the_logger,
            the_audit=the_audit,
        )

    assert exc.value.status_code == 403
    assert exc.value.detail == loggers.AUTHZ_ADMIN_ACCESS_REQUIRED
    the_installation.reload_room_configs.assert_not_awaited()
    the_audit.rooms_reloaded.assert_not_called()


@pytest.mark.anyio
async def test_reload_rooms_configs_multi_writer(
    monkeypatch,
):
    """A multi-writer deployment answers 409 with no in-process swap."""
    monkeypatch.setenv("_SOLIPLEX_MULTIPLE_WRITERS", "Y")

    i_config = mock.create_autospec(config_installation.InstallationConfig)
    the_installation = installation.Installation(i_config)
    the_installation.reload_room_configs = mock.AsyncMock()
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = True
    the_logger = mock.create_autospec(loggers.LogWrapper)
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    with pytest.raises(fastapi.HTTPException) as exc:
        await installation_views.reload_rooms_configs(
            the_installation=the_installation,
            the_admin_users=the_admin_users,
            the_user_claims=THE_USER_CLAIMS,
            the_logger=the_logger,
            the_audit=the_audit,
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == loggers.INST_RELOAD_ROOMS_MULTI_WRITER
    the_installation.reload_room_configs.assert_not_awaited()
    the_audit.rooms_reloaded.assert_not_called()


@pytest.mark.parametrize(
    "admin, writer, status",
    [(False, None, 403), (True, "Y", 409), (True, "", 409)],
)
def test_reload_skills_configs_refused(monkeypatch, admin, writer, status):
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)
    if writer is not None:
        monkeypatch.setenv("_SOLIPLEX_MULTIPLE_WRITERS", writer)
    the_installation = mock.create_autospec(installation.Installation)
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = admin
    the_logger = mock.create_autospec(loggers.LogWrapper)
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)
    app = _skills_reload_app(
        the_installation, the_admin_users, the_logger, the_audit
    )

    with TestClient(app) as client:
        response = client.post("/v1/installation/reload/skills")

    assert response.status_code == status
    assert response.json() == {
        "detail": (
            loggers.AUTHZ_ADMIN_ACCESS_REQUIRED
            if status == 403
            else loggers.INST_RELOAD_SKILLS_MULTI_WRITER
        )
    }
    the_installation.reload_skill_and_room_configs.assert_not_awaited()
    the_audit.skills_reloaded.assert_not_called()


def _skills_reload_app(
    the_installation, the_admin_users, the_logger, the_audit
):
    app = fastapi.FastAPI()
    app.include_router(installation_views.router)
    app.dependency_overrides = {
        installation_views.depend_the_installation.dependency: (
            lambda: the_installation
        ),
        installation_views.depend_the_admin_users.dependency: (
            lambda: the_admin_users
        ),
        installation_views.depend_the_user_claims.dependency: (
            lambda: THE_USER_CLAIMS
        ),
        installation_views.depend_the_logger.dependency: lambda: the_logger,
        installation_views.get_the_config_audit_log: lambda: the_audit,
    }
    return app


def _write_skill(root, name, instructions="Original instructions."):
    directory = root / "skills" / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: A test skill.\n---\n"
        f"{instructions}\n",
        encoding="utf-8",
    )
    return path


def _write_room(root, name="base-room", bindings=("base",), native=None):
    directory = root / "rooms" / name
    directory.mkdir(parents=True, exist_ok=True)
    room = {
        "id": name,
        "name": name,
        "description": "A test room.",
        "agent": {"model_name": "test-model", "system_prompt": "Help."},
        "skills": {"installation_skill_names": list(bindings)},
    }
    if native:
        room["skills"]["skill_configs"] = [{"kind": native}]
    path = directory / "room_config.yaml"
    path.write_text(yaml.safe_dump(room), encoding="utf-8")
    return path


@pytest.fixture
def loaded_config(tmp_path):
    _write_skill(tmp_path, "base")
    _write_room(tmp_path)
    config = config_installation.InstallationConfig(
        id="reload-defects", _config_path=tmp_path / "installation.yaml"
    )
    assert set(config.room_configs["base-room"].skill_configs) == {"base"}
    return config


def _loaded_skills_reload_app(config):
    admin = mock.create_autospec(authz.AdminUserPolicy)
    admin.check_admin_access.return_value = True
    return _skills_reload_app(
        installation.Installation(config),
        admin,
        mock.create_autospec(loggers.LogWrapper),
        mock.create_autospec(loggers.InstallationConfigAuditLog),
    )


def test_endpoint_unchanged_native_skill_has_empty_delta(
    loaded_config, tmp_path, monkeypatch
):
    """Comparing native skills avoids recursion through room snapshots."""
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)
    _write_room(tmp_path, native="haiku.rag.skills.evidence_compaction")
    loaded_config.reload_skill_and_room_configs()

    with TestClient(_loaded_skills_reload_app(loaded_config)) as client:
        response = client.post("/v1/installation/reload/skills")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "rooms": {"added": [], "removed": [], "changed": []},
        "skills": {"added": [], "removed": [], "changed": []},
    }


def test_endpoint_ignores_room_quiz_cache_in_delta(loaded_config, monkeypatch):
    """Reading a derived room cache preserves an empty reload delta."""
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)
    assert loaded_config.room_configs["base-room"].quiz_map == {}

    with TestClient(_loaded_skills_reload_app(loaded_config)) as client:
        response = client.post("/v1/installation/reload/skills")

    assert response.status_code == 200
    assert response.json()["rooms"]["changed"] == []


def test_concurrent_endpoint_deltas_use_their_own_skill_generation(
    loaded_config, tmp_path, monkeypatch
):
    """Each concurrent reload reports its own skill generation."""
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)
    _write_skill(tmp_path, "first")
    _write_room(tmp_path, "first-room", ("first",))
    both_started = threading.Barrier(2)
    first_published = threading.Event()
    second_published = threading.Event()
    counter_lock = threading.Lock()
    calls = 0
    reload_configs = type(loaded_config).reload_skill_and_room_configs

    def scheduled_reload(config):
        nonlocal calls
        with counter_lock:
            calls += 1
            number = calls
        both_started.wait(timeout=10)
        if number == 1:
            snapshots = reload_configs(config)
            first_published.set()
            assert second_published.wait(timeout=10)
        else:
            assert first_published.wait(timeout=10)
            (tmp_path / "skills" / "first" / "SKILL.md").unlink()
            (tmp_path / "rooms" / "first-room" / "room_config.yaml").unlink()
            _write_skill(tmp_path, "second")
            _write_room(tmp_path, "second-room", ("second",))
            try:
                snapshots = reload_configs(config)
            finally:
                second_published.set()
        return snapshots

    monkeypatch.setattr(
        type(loaded_config), "reload_skill_and_room_configs", scheduled_reload
    )
    with TestClient(_loaded_skills_reload_app(loaded_config)) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = [
                executor.submit(client.post, "/v1/installation/reload/skills")
                for _ in range(2)
            ]
            payloads = [
                future.result(timeout=20).json() for future in responses
            ]

    (first_delta,) = [
        delta
        for delta in payloads
        if delta["rooms"]["added"] == ["first-room"]
    ]
    assert first_delta["skills"]["added"] == ["first"]


@pytest.mark.parametrize("empty", [False, True])
def test_reload_skills_configs(monkeypatch, audit_records, empty):
    monkeypatch.delenv("_SOLIPLEX_MULTIPLE_WRITERS", raising=False)
    added = () if empty else ("new-skill",)
    removed = () if empty else ("old-skill",)
    changed = () if empty else ("changed-skill",)
    rooms_added = () if empty else ("new-room",)
    rooms_removed = () if empty else ("old-room",)
    rooms_changed = () if empty else ("changed-room",)
    the_installation = mock.create_autospec(installation.Installation)
    the_installation.reload_skill_and_room_configs.return_value = (
        installation.SkillAndRoomConfigsReloaded(
            rooms=installation.RoomConfigsReloaded(
                added=rooms_added,
                removed=rooms_removed,
                changed=rooms_changed,
            ),
            skills_added=added,
            skills_removed=removed,
            skills_changed=changed,
        )
    )
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = True
    the_logger = mock.create_autospec(loggers.LogWrapper)
    the_audit = loggers.InstallationConfigAuditLog(claims=THE_USER_CLAIMS)
    app = _skills_reload_app(
        the_installation, the_admin_users, the_logger, the_audit
    )

    with TestClient(app) as client:
        response = client.post("/v1/installation/reload/skills")

    skills_delta = {
        "added": list(added),
        "removed": list(removed),
        "changed": list(changed),
    }
    rooms_delta = {
        "added": list(rooms_added),
        "removed": list(rooms_removed),
        "changed": list(rooms_changed),
    }
    assert response.status_code == 200
    assert response.json() == {"rooms": rooms_delta, "skills": skills_delta}
    the_installation.reload_skill_and_room_configs.assert_awaited_once_with()
    the_admin_users.check_admin_access.assert_awaited_once_with(
        THE_USER_CLAIMS,
        resource=loggers.AUDIT_RESOURCE_INSTALLATION_SKILLS,
        action=loggers.AUDIT_ACTION_UPDATE,
    )
    the_logger.bind.assert_called_once_with(loggers.AUTHZ_LOGGER_NAME)
    the_logger.bind.return_value.debug.assert_called_once_with(
        loggers.INST_POST_RELOAD_SKILLS
    )
    record = audit_records[-1]
    assert record.getMessage() == loggers.AUDIT_INSTALLATION_SKILLS_RELOADED
    assert record.outcome == loggers.AUDIT_OUTCOME_SUCCESS
    assert record.claims == THE_USER_CLAIMS
    assert record.rooms_added == list(rooms_added)
    assert record.rooms_removed == list(rooms_removed)
    assert record.rooms_changed == list(rooms_changed)
    assert record.skills_added == list(added)
    assert record.skills_removed == list(removed)
    assert record.skills_changed == list(changed)


@pytest.mark.anyio
@mock.patch("soliplex.views.installation.subprocess")
async def test_get_installation_versions_w_error(sp):
    sp.check_output.side_effect = ValueError("testing")

    the_installation = object()
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = True
    the_logger = mock.create_autospec(loggers.LogWrapper)
    bound_logger = the_logger.bind.return_value
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    def _check(exc):
        return exc.status_code == 500

    with pytest.raises(fastapi.HTTPException, check=_check) as exc:
        await installation_views.get_installation_versions(
            the_installation=the_installation,
            the_admin_users=the_admin_users,
            the_user_claims=THE_USER_CLAIMS,
            the_logger=the_logger,
            the_audit=the_audit,
        )

    assert exc.value.detail == loggers.INST_SUBPROCESS_PIP

    the_admin_users.check_admin_access.assert_awaited_once_with(
        THE_USER_CLAIMS,
        resource=loggers.AUDIT_RESOURCE_INSTALLATION_VERSIONS,
        action=loggers.AUDIT_ACTION_READ,
    )

    # The read is audited once the admin gate passes, before the
    # subprocess (which here fails with a 500).
    the_audit.installation_versions_read.assert_called_once_with()
    bound_logger.exception.assert_called_once_with(
        loggers.INST_SUBPROCESS_PIP,
    )
    bound_logger.debug.assert_called_once_with(
        loggers.INST_GET_INSTALLATION_VERSIONS,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("w_admin_access", [False, True])
@mock.patch("soliplex.views.installation.subprocess")
async def test_get_installation_versions_wo_error(sp, w_admin_access):
    pip_manifest = [
        {"name": "one-package", "version": "0.1.2"},
        {"name": "another-package", "version": "3.4.5", "extra": "blather"},
    ]
    expected = {
        "one-package": {
            "version": "0.1.2",
        },
        "another-package": {
            "version": "3.4.5",
            "extra": "blather",
        },
    }
    sp.check_output.return_value = json.dumps(pip_manifest).encode("utf8")

    the_installation = object()
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = w_admin_access
    the_logger = mock.create_autospec(loggers.LogWrapper)
    bound_logger = the_logger.bind.return_value
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    if not w_admin_access:
        with pytest.raises(fastapi.HTTPException) as exc:
            await installation_views.get_installation_versions(
                the_installation=the_installation,
                the_admin_users=the_admin_users,
                the_user_claims=THE_USER_CLAIMS,
                the_logger=the_logger,
                the_audit=the_audit,
            )

        assert exc.value.status_code == 403
        assert exc.value.detail == loggers.AUTHZ_ADMIN_ACCESS_REQUIRED
        the_audit.installation_versions_read.assert_not_called()

    else:
        found = await installation_views.get_installation_versions(
            the_installation=the_installation,
            the_admin_users=the_admin_users,
            the_user_claims=THE_USER_CLAIMS,
            the_logger=the_logger,
            the_audit=the_audit,
        )

        assert found == expected
        the_audit.installation_versions_read.assert_called_once_with()

    the_admin_users.check_admin_access.assert_awaited_once_with(
        THE_USER_CLAIMS,
        resource=loggers.AUDIT_RESOURCE_INSTALLATION_VERSIONS,
        action=loggers.AUDIT_ACTION_READ,
    )

    bound_logger.debug.assert_called_once_with(
        loggers.INST_GET_INSTALLATION_VERSIONS,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("w_admin_access", [False, True])
async def test_get_installation_providers(w_admin_access):
    PROVIDER_INFO = {
        config_agents.LLMProviderType.OLLAMA: {
            OLLAMA_BASE_URL: {
                TEST_MODEL_ONE: installation.ProviderRole.CHAT,
                TEST_MODEL_TWO: installation.ProviderRole.EMBEDDING,
            },
        }
    }

    the_installation = mock.create_autospec(installation.Installation)
    the_installation.all_provider_info = PROVIDER_INFO
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = w_admin_access
    the_logger = mock.create_autospec(loggers.LogWrapper)
    bound_logger = the_logger.bind.return_value
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    if not w_admin_access:
        with pytest.raises(fastapi.HTTPException) as exc:
            await installation_views.get_installation_providers(
                the_installation=the_installation,
                the_admin_users=the_admin_users,
                the_user_claims=THE_USER_CLAIMS,
                the_logger=the_logger,
                the_audit=the_audit,
            )

        assert exc.value.status_code == 403
        assert exc.value.detail == loggers.AUTHZ_ADMIN_ACCESS_REQUIRED
        the_audit.installation_providers_read.assert_not_called()

    else:
        found = await installation_views.get_installation_providers(
            the_installation=the_installation,
            the_admin_users=the_admin_users,
            the_user_claims=THE_USER_CLAIMS,
            the_logger=the_logger,
            the_audit=the_audit,
        )

        assert found == PROVIDER_INFO
        the_audit.installation_providers_read.assert_called_once_with()

    the_admin_users.check_admin_access.assert_awaited_once_with(
        THE_USER_CLAIMS,
        resource=loggers.AUDIT_RESOURCE_INSTALLATION_PROVIDERS,
        action=loggers.AUDIT_ACTION_READ,
    )

    bound_logger.debug.assert_called_once_with(
        loggers.INST_GET_INSTALLATION_PROVIDERS,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("w_admin_access", [False, True])
@mock.patch("soliplex.util.GitMetadata")
async def test_get_installation_git_metadata(gm_klass, w_admin_access):
    gm = gm_klass.return_value
    gm.git_hash = GIT_HASH
    gm.git_branch = GIT_BRANCH
    gm.git_tag = GIT_TAG

    the_installation = mock.create_autospec(installation.Installation)
    the_admin_users = mock.create_autospec(authz.AdminUserPolicy)
    the_admin_users.check_admin_access.return_value = w_admin_access
    the_logger = mock.create_autospec(loggers.LogWrapper)
    bound_logger = the_logger.bind.return_value
    the_audit = mock.create_autospec(loggers.InstallationConfigAuditLog)

    if not w_admin_access:
        with pytest.raises(fastapi.HTTPException) as exc:
            await installation_views.get_installation_git_metadata(
                the_installation=the_installation,
                the_admin_users=the_admin_users,
                the_user_claims=THE_USER_CLAIMS,
                the_logger=the_logger,
                the_audit=the_audit,
            )

        assert exc.value.status_code == 403
        assert exc.value.detail == loggers.AUTHZ_ADMIN_ACCESS_REQUIRED
        the_audit.installation_git_metadata_read.assert_not_called()

    else:
        found = await installation_views.get_installation_git_metadata(
            the_installation=the_installation,
            the_admin_users=the_admin_users,
            the_user_claims=THE_USER_CLAIMS,
            the_logger=the_logger,
            the_audit=the_audit,
        )

        assert found == models.GitMetadata(
            git_hash=GIT_HASH,
            git_branch=GIT_BRANCH,
            git_tag=GIT_TAG,
        )
        the_audit.installation_git_metadata_read.assert_called_once_with()

    bound_logger.debug.assert_called_once_with(
        loggers.INST_GET_INSTALLATION_GIT_METADATA,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "server_name,server_description",
    [
        ("Demo Server", "A friendly demo instance"),
        ("Demo Server", None),
        (None, "A friendly demo instance"),
    ],
)
async def test_get_installation_identity(server_name, server_description):
    i_config = mock.create_autospec(config_installation.InstallationConfig)
    i_config.id = "soliplex-conf-minimal"
    i_config.server_name = server_name
    i_config.server_description = server_description
    the_installation = installation.Installation(i_config)
    the_unauth_logger = mock.create_autospec(loggers.LogWrapper)

    found = await installation_views.get_installation_identity(
        the_installation=the_installation,
        the_unauth_logger=the_unauth_logger,
    )

    assert isinstance(found, models.ServerInfo)
    assert found.installation_id == "soliplex-conf-minimal"
    assert found.name == server_name
    assert found.description == server_description
    the_unauth_logger.debug.assert_called_once_with(
        loggers.INST_GET_INSTALLATION_IDENTITY,
    )


@pytest.mark.anyio
async def test_get_installation_identity_unconfigured_returns_404():
    i_config = mock.create_autospec(config_installation.InstallationConfig)
    i_config.server_name = None
    i_config.server_description = None
    the_installation = installation.Installation(i_config)
    the_unauth_logger = mock.create_autospec(loggers.LogWrapper)

    with pytest.raises(fastapi.HTTPException) as exc:
        await installation_views.get_installation_identity(
            the_installation=the_installation,
            the_unauth_logger=the_unauth_logger,
        )

    assert exc.value.status_code == 404
    the_unauth_logger.debug.assert_called_once_with(
        loggers.INST_NO_INSTALLATION_IDENTITY,
    )
