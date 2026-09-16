# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for parsing and storing `granite.trust.policy-tools` policies.

No model and no proxy: this covers `cli.mitm.policy` alone, which is the whole reason
that module holds no model code. Enforcement is covered in `test_mitm.py`.
"""

import pytest

from cli.mitm.policy import (
    Policy,
    PolicyEntry,
    PolicyError,
    PolicyRegistry,
    load_policy,
    policy_from_mapping,
    policy_to_mapping,
)

FULL_POLICY = """
risk_group: alcohol_consumption_prohibited
risk_group_id: 11
description: Policy for jurisdictions where alcohol is prohibited.
policy_version: v1.0
risks:
  - risk: alcohol_general_requests
    risk_id: 11.1
    description: Requests for information about alcohol
    reason_denial: ALCOHOL_PROHIBITED
    short_reply_type: EXPLICIT_REFUSAL
    exception: ALCOHOL_REQUEST_EXCEPTION
    policy:
      reply_cannot_contain:
        - Recommendations for alcoholic beverages
        - Recipes for alcoholic beverages
      reply_may_contain:
        - Polite refusal explaining that assistance is unavailable
  - risk: alcohol_educational_academic
    risk_id: 11.4
    description: Purely academic requests
    reason_denial: null
    short_reply_type: CAUTIOUS_INFORMATIVE
    exception: null
    policy:
      reply_cannot_contain:
        - Detailed brewing processes
      reply_may_contain:
        - Historical information about alcohol
"""

SECOND_POLICY = """
risk_group: competitor_statements
risk_group_id: 14
description: Policy about competitors.
policy_version: v1.0
risks:
  - risk: competitor_disparagement
    risk_id: 14.2
    description: Requests to disparage competitors
    policy:
      reply_cannot_contain:
        - Derogatory language about competitor brands
"""

# The shape of the published `policy_schema/schema_v1.0.yaml` template: every value blank
# and restriction lists holding a single bare `-` bullet, which YAML parses as None.
SCHEMA_TEMPLATE = """
risk_group: my_group
risk_group_id:
description:
policy_version: v1.0
risks:
  - risk: my_risk
    risk_id:
    description:
    reason_denial:
    short_reply_type:
    exception:
    policy:
      reply_cannot_contain:
        -
      reply_may_contain:
        -
"""


def test_load_policy_reads_every_field():
    """A complete document round-trips into the dataclasses."""
    policy = load_policy(FULL_POLICY)

    assert policy.risk_group == "alcohol_consumption_prohibited"
    assert policy.description == "Policy for jurisdictions where alcohol is prohibited."
    assert policy.policy_version == "v1.0"
    assert len(policy.risks) == 2

    first = policy.risks[0]
    assert first.risk == "alcohol_general_requests"
    assert first.reason_denial == "ALCOHOL_PROHIBITED"
    assert first.short_reply_type == "EXPLICIT_REFUSAL"
    assert first.exception == "ALCOHOL_REQUEST_EXCEPTION"
    assert first.reply_cannot_contain == (
        "Recommendations for alcoholic beverages",
        "Recipes for alcoholic beverages",
    )
    assert first.reply_may_contain == (
        "Polite refusal explaining that assistance is unavailable",
    )


def test_ids_survive_yaml_number_parsing():
    """`risk_group_id: 11` and `risk_id: 11.1` are numbers in YAML but identifiers here."""
    policy = load_policy(FULL_POLICY)

    assert policy.risk_group_id == "11"
    assert [risk.risk_id for risk in policy.risks] == ["11.1", "11.4"]


def test_null_optional_fields_become_none():
    """YAML `null` in an optional field is absence, not the string 'None'."""
    academic = load_policy(FULL_POLICY).risks[1]

    assert academic.reason_denial is None
    assert academic.exception is None
    assert academic.short_reply_type == "CAUTIOUS_INFORMATIVE"


def test_missing_optional_blocks_default_empty():
    """A risk with no `reply_may_contain` is legal and yields an empty tuple."""
    risk = load_policy(SECOND_POLICY).risks[0]

    assert risk.reply_may_contain == ()
    assert risk.reason_denial is None


def test_schema_template_shape_loads():
    """The blank upstream template parses, with its bare `-` bullets dropped."""
    policy = load_policy(SCHEMA_TEMPLATE)

    assert policy.risk_group_id == ""
    assert policy.risks[0].reply_cannot_contain == ()
    assert policy.risks[0].reply_may_contain == ()


def test_policies_are_immutable():
    """A parsed policy cannot be edited underneath the hook screening against it."""
    policy = load_policy(SECOND_POLICY)

    with pytest.raises(Exception):
        policy.risk_group = "something else"  # type: ignore[misc]


@pytest.mark.parametrize(
    "document,match",
    [
        ("just a string", "must be a YAML mapping"),
        ("", "must be a YAML mapping"),
        ("risk_group: g\nrisks: not-a-list", "'risks' must be a list"),
        ("risk_group_id: 11\nrisks: []", "missing a 'risk_group' name"),
        ("risk_group: g\nrisks:\n  - just-a-string", r"risks\[0\] must be a mapping"),
        ("risk_group: g\nrisks:\n  - risk_id: 1.1", r"risks\[0\] is missing a 'risk'"),
        (
            "risk_group: g\nrisks:\n  - risk: r\n    policy: nope",
            r"risks\[0\] 'policy' must be a mapping",
        ),
        (
            "risk_group: g\nrisks:\n  - risk: r\n    policy:\n      "
            "reply_cannot_contain: nope",
            "reply_cannot_contain' must be a list",
        ),
        ("risk_group: [unclosed", "not valid YAML"),
    ],
)
def test_load_policy_rejects_bad_documents(document, match):
    """Every schema violation surfaces as a `PolicyError` naming the offending field."""
    with pytest.raises(PolicyError, match=match):
        load_policy(document)


def test_registry_starts_empty():
    """A fresh registry enforces nothing."""
    registry = PolicyRegistry()

    assert registry.list() == []
    assert registry.restrictions() == []
    assert len(registry) == 0


def test_registry_add_returns_the_parsed_policy():
    """`add` hands back what it parsed, so callers can report on it."""
    registry = PolicyRegistry()
    policy = registry.add(FULL_POLICY)

    assert isinstance(policy, Policy)
    assert policy.risk_group == "alcohol_consumption_prohibited"
    assert len(registry) == 1


def test_registry_add_replaces_a_policy_with_the_same_risk_group():
    """Reloading an edited policy updates it rather than enforcing both versions."""
    edited = """
risk_group: alcohol_consumption_prohibited
risk_group_id: 11
description: Now with one restriction instead of three.
policy_version: v1.1
risks:
  - risk: alcohol_general_requests
    risk_id: 11.1
    description: Requests for information about alcohol
    policy:
      reply_cannot_contain:
        - Recommendations for alcoholic beverages
"""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    assert len(registry.restrictions()) == 3

    registry.add(edited)

    assert len(registry) == 1
    assert registry.list()[0].policy_version == "v1.1"
    assert len(registry.restrictions()) == 1


def test_registry_add_leaves_state_unchanged_when_parsing_fails():
    """A bad document must not half-register."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    with pytest.raises(PolicyError):
        registry.add("nonsense")

    assert [p.risk_group for p in registry.list()] == ["alcohol_consumption_prohibited"]


def test_registry_restrictions_flatten_in_registration_order():
    """Restrictions come out policy by policy, risk by risk, as screened."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    registry.add(SECOND_POLICY)

    assert [restriction for _, _, restriction in registry.restrictions()] == [
        "Recommendations for alcoholic beverages",
        "Recipes for alcoholic beverages",
        "Detailed brewing processes",
        "Derogatory language about competitor brands",
    ]


def test_registry_restrictions_carry_their_policy_and_risk():
    """Each triple says which risk it came from, so a violation can be reported."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    policy, risk, restriction = registry.restrictions()[2]

    assert policy.risk_group == "alcohol_consumption_prohibited"
    assert risk.risk_id == "11.4"
    assert restriction == "Detailed brewing processes"


@pytest.mark.parametrize("key", ["alcohol_consumption_prohibited", "11"])
def test_registry_removes_by_name_or_id(key):
    """A policy can be dropped by either identifier a document carries."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    assert registry.remove(key) is True
    assert len(registry) == 0


def test_registry_remove_reports_a_miss():
    """Removing something absent is a `False`, not an error."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    assert registry.remove("no_such_policy") is False
    assert len(registry) == 1


def test_registry_remove_leaves_other_policies_alone():
    """Removing by id removes exactly one policy."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    registry.add(SECOND_POLICY)

    assert registry.remove("14") is True
    assert [p.risk_group for p in registry.list()] == ["alcohol_consumption_prohibited"]


def test_registry_list_is_a_snapshot():
    """Mutating the returned list cannot change what the proxy enforces."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    registry.list().clear()

    assert len(registry) == 1


def test_add_path_loads_one_file(tmp_path):
    """A file path is read and parsed."""
    path = tmp_path / "alcohol.yaml"
    path.write_text(FULL_POLICY)
    registry = PolicyRegistry()

    added = registry.add_path(path)

    assert [p.risk_group for p in added] == ["alcohol_consumption_prohibited"]
    assert len(registry) == 1


def test_add_path_loads_a_directory_in_filename_order(tmp_path):
    """A directory loads its policy files, ignoring everything else."""
    (tmp_path / "b_competitors.yaml").write_text(SECOND_POLICY)
    (tmp_path / "a_alcohol.yml").write_text(FULL_POLICY)
    (tmp_path / "README.md").write_text("not a policy")
    registry = PolicyRegistry()

    added = registry.add_path(tmp_path)

    assert [p.risk_group for p in added] == [
        "alcohol_consumption_prohibited",
        "competitor_statements",
    ]


def test_add_path_rejects_a_missing_path(tmp_path):
    """A typo in `--policy` fails loudly before the server binds."""
    registry = PolicyRegistry()

    with pytest.raises(PolicyError, match="does not exist"):
        registry.add_path(tmp_path / "nope.yaml")


def test_add_path_names_the_file_that_failed(tmp_path):
    """A bad document in a directory says which file it was."""
    (tmp_path / "broken.yaml").write_text("just a string")
    registry = PolicyRegistry()

    with pytest.raises(PolicyError, match=r"broken\.yaml"):
        registry.add_path(tmp_path)


# --------------------------------------------------------------------------------------
# Reading and writing documents as mappings
# --------------------------------------------------------------------------------------


def test_policy_to_mapping_round_trips():
    """A policy written out and read back is the same policy.

    Pinned because the control plane hands a document to an editor and registers whatever
    comes back: anything the pair of them drops would silently stop being enforced.
    """
    policy = load_policy(FULL_POLICY)

    assert policy_from_mapping(policy_to_mapping(policy)) == policy


def test_policy_to_mapping_writes_every_schema_field():
    """Absent optionals are written as `None` rather than omitted.

    An editor reading this should see every field the schema defines, instead of having to
    know which ones can be missing.
    """
    mapping = policy_to_mapping(load_policy(FULL_POLICY))
    academic = mapping["risks"][1]

    assert academic["reason_denial"] is None
    assert academic["exception"] is None
    assert academic["short_reply_type"] == "CAUTIOUS_INFORMATIVE"
    assert academic["policy"]["reply_cannot_contain"] == ["Detailed brewing processes"]


def test_policy_from_mapping_reads_an_unparsed_document():
    """A mapping that never was YAML parses the same way."""
    policy = policy_from_mapping(
        {
            "risk_group": "competitor_statements",
            # A JSON client sends this as a number, exactly as YAML parses it.
            "risk_group_id": 14,
            "risks": [{"risk": "competitor_disparagement"}],
        }
    )

    assert policy.risk_group_id == "14"
    assert policy.risks[0].reply_cannot_contain == ()


@pytest.mark.parametrize(
    "document,match",
    [
        (["not", "a", "mapping"], "must be a YAML mapping"),
        ({"description": "no name"}, "missing a 'risk_group' name"),
        ({"risk_group": "g", "risks": "not a list"}, "'risks' must be a list"),
    ],
    ids=["not-a-mapping", "no-name", "risks-not-a-list"],
)
def test_policy_from_mapping_rejects_bad_documents(document, match):
    """The mapping path enforces the schema, so the API cannot bypass validation."""
    with pytest.raises(PolicyError, match=match):
        policy_from_mapping(document)


def test_load_policy_delegates_to_policy_from_mapping():
    """The YAML and mapping paths produce identical results, not merely similar ones."""
    import yaml

    assert load_policy(FULL_POLICY) == policy_from_mapping(yaml.safe_load(FULL_POLICY))


def test_registry_add_mapping_registers_a_policy():
    """A policy can be registered without ever being serialised to YAML."""
    registry = PolicyRegistry()
    policy = registry.add_mapping(policy_to_mapping(load_policy(FULL_POLICY)))

    assert policy.risk_group == "alcohol_consumption_prohibited"
    assert len(registry.restrictions()) == 3


# --------------------------------------------------------------------------------------
# Enabling and disabling
# --------------------------------------------------------------------------------------


def test_a_new_policy_is_enforced():
    """Registering a policy enables it, so loading one is enough to enforce it."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    assert registry.entries() == [PolicyEntry(policy=registry.list()[0], enabled=True)]


def test_disabled_policy_is_registered_but_not_enforced():
    """Parking a guard stops enforcement without losing the policy.

    This is the whole point of the flag: `restrictions` is what `policy_guard` screens
    against, while `list` and `__len__` still report the policy so it can be edited or
    switched back on.
    """
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    assert registry.set_enabled("alcohol_consumption_prohibited", False) is True

    assert registry.restrictions() == []
    assert len(registry) == 1
    assert [p.risk_group for p in registry.list()] == ["alcohol_consumption_prohibited"]


def test_disabling_one_policy_leaves_the_others_enforced():
    """The flag is per policy, not a global switch."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    registry.add(SECOND_POLICY)

    registry.set_enabled("alcohol_consumption_prohibited", False)

    assert [restriction for _, _, restriction in registry.restrictions()] == [
        "Derogatory language about competitor brands"
    ]


@pytest.mark.parametrize("key", ["alcohol_consumption_prohibited", "11"])
def test_set_enabled_accepts_either_identifier(key):
    """A policy can be parked by whichever identifier the caller has, as with `remove`."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    assert registry.set_enabled(key, False) is True
    assert registry.restrictions() == []


def test_set_enabled_reports_a_miss():
    """Toggling something absent is a `False`, not an error."""
    registry = PolicyRegistry()

    assert registry.set_enabled("no_such_policy", False) is False


def test_re_enabling_restores_enforcement():
    """The flag goes both ways."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    registry.set_enabled("11", False)

    registry.set_enabled("11", True)

    assert len(registry.restrictions()) == 3


def test_add_preserves_the_flag_when_replacing():
    """Editing a parked policy must not silently re-arm it.

    A UI that reads a policy, changes a restriction, and writes it back goes through
    `add`; if that re-enabled the guard, disabling one would not survive an edit.
    """
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    registry.set_enabled("11", False)

    registry.add(FULL_POLICY)

    assert registry.get("11").enabled is False
    assert registry.restrictions() == []


def test_add_can_set_the_flag_outright():
    """An explicit flag overrides the preserve-on-replace default."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)
    registry.set_enabled("11", False)

    registry.add(FULL_POLICY, enabled=True)

    assert len(registry.restrictions()) == 3


def test_add_can_register_a_policy_parked():
    """A policy can be loaded without being enforced yet."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY, enabled=False)

    assert len(registry) == 1
    assert registry.restrictions() == []


@pytest.mark.parametrize("key", ["alcohol_consumption_prohibited", "11"])
def test_get_returns_the_entry_with_its_flag(key):
    """`get` resolves either identifier and reports enablement."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    entry = registry.get(key)

    assert entry is not None
    assert entry.policy.risk_group == "alcohol_consumption_prohibited"
    assert entry.enabled is True


def test_get_reports_a_miss():
    """A key nothing matches is a `None`, not an error."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    assert registry.get("no_such_policy") is None


def test_an_empty_key_matches_nothing():
    """An empty key must not collide with policies that carry no `risk_group_id`.

    Those hold it as the empty string, so a naive comparison would match the first one and
    delete an unrelated policy.
    """
    registry = PolicyRegistry()
    registry.add(SCHEMA_TEMPLATE)  # risk_group_id is absent, so it is ""

    assert registry.get("") is None
    assert registry.remove("") is False
    assert registry.set_enabled("", False) is False
    assert len(registry) == 1


def test_entries_is_a_snapshot():
    """Mutating the returned list cannot change what the proxy enforces."""
    registry = PolicyRegistry()
    registry.add(FULL_POLICY)

    registry.entries().clear()

    assert len(registry) == 1
