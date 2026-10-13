"""The protonation source: parsing, per-site reference pKa, and the engine helpers."""

import dataclasses
import logging

import pytest
from objective_fixtures import NAMES, PARENTS, SPEC
from test_session import make_config, make_inputs

from mpnn.ph.binding import DEFAULT_MODEL_PKA, linked_sites_from_spec
from mpnn.ph.config import ProtonationSource, SwitchDesignConfig
from mpnn.ph.session import run_switch_design
from mpnn.ph.states import SiteKey, StateSpec, site_index_from_arrays
from mpnn.ph.vocab_meta import TokenTable

A1, B2 = SiteKey("A", 1), SiteKey("B", 2)  # the two titratable sites of the toy complex
LINKED = dict(binding_model="linked_equilibrium", binding_options={"beta": 0.7})


def test_ev6_is_the_default_and_takes_no_ph():
    assert ProtonationSource().source == "ev6"
    assert make_config().protonation == ProtonationSource()
    with pytest.raises(ValueError, match="only apply to the 'propka' source"):
        ProtonationSource(source="ev6", ph=7.4)
    with pytest.raises(ValueError, match="only apply to the 'propka' source"):
        ProtonationSource(source="ev6", margin=0.5)


def test_propka_needs_a_ph_and_has_a_default_margin():
    source = ProtonationSource.from_dict({"source": "propka", "ph": 7.4})
    assert (source.ph, source.margin) == (7.4, 0.5)
    assert ProtonationSource("propka", 6.0, 0.0).margin == 0.0
    for bad in ({"source": "propka"}, {"source": "propka", "ph": float("nan")}):
        with pytest.raises(ValueError, match="finite protonation.ph"):
            ProtonationSource.from_dict(bad)
    with pytest.raises(ValueError, match="margin"):
        ProtonationSource.from_dict({"source": "propka", "ph": 7.0, "margin": -0.1})
    with pytest.raises(ValueError, match="must be one of"):
        ProtonationSource.from_dict({"source": "pkacalc"})
    with pytest.raises(ValueError, match="Unknown protonation keys"):
        ProtonationSource.from_dict({"source": "propka", "ph": 7.0, "pH": 7.0})


def test_the_config_parses_the_protonation_section():
    config = make_config(protonation={"source": "propka", "ph": 6.5})
    assert config.protonation == ProtonationSource("propka", 6.5, 0.5)
    with pytest.raises(ValueError, match="Unknown protonation keys"):
        make_config(protonation={"sauce": "propka"})


def sites(by_site=None, by_parent=None):
    spec = StateSpec.from_dict(SPEC)
    index = site_index_from_arrays(["A", "A", "A", "B", "B", "B"], [1, 2, 3, 1, 2, 3])
    found = linked_sites_from_spec(
        spec, TokenTable(NAMES), index, PARENTS, by_parent, pka_by_site=by_site
    )
    return {site.position: site.pka for site in found}


def test_a_sites_own_pka_beats_the_value_for_its_residue_type(caplog):
    assert sites() == {0: DEFAULT_MODEL_PKA["HIS"], 4: DEFAULT_MODEL_PKA["HIS"]}
    assert sites(by_parent={"HIS": 6.0}) == {0: 6.0, 4: 6.0}
    assert sites(by_site={A1: 5.2, B2: 6.9}, by_parent={"HIS": 6.0}) == {0: 5.2, 4: 6.9}
    with caplog.at_level(logging.WARNING):  # one site lacks its own value
        assert sites(by_site={A1: 5.2}, by_parent={"HIS": 6.0}) == {0: 5.2, 4: 6.0}
    assert "No per-site pKa for B:2" in caplog.text
    with pytest.raises(ValueError, match="must be finite"):
        sites(by_site={A1: float("nan"), B2: 6.9})


def chosen_pka(run):
    return {site.position: site.pka for site in run.objective.ctx.binding.sites}


def test_propka_values_reach_the_linked_equilibrium_model():
    inputs = dataclasses.replace(make_inputs(), site_pka={A1: 5.2, B2: 6.9})
    propka = {"source": "propka", "ph": 7.4}
    run = run_switch_design(inputs, make_config(protonation=propka, **LINKED))
    assert chosen_pka(run) == {0: 5.2, 4: 6.9}
    # the user's own value for one site wins over PROPKA's
    explicit = dict(LINKED, binding_options={"beta": 0.7, "pka_by_site": {"B:2": 4.0}})
    run = run_switch_design(inputs, make_config(protonation=propka, **explicit))
    assert chosen_pka(run) == {0: 5.2, 4: 4.0}
    # without the propka source the per-type constants stay in force
    run = run_switch_design(inputs, make_config(**LINKED))
    assert chosen_pka(run) == {0: DEFAULT_MODEL_PKA["HIS"], 4: DEFAULT_MODEL_PKA["HIS"]}


def test_propka_pka_values_are_required_when_propka_is_the_source():
    propka = {"source": "propka", "ph": 7.4}
    with pytest.raises(ValueError, match="DesignInputs.site_pka"):
        run_switch_design(make_inputs(), make_config(protonation=propka, **LINKED))
    # the state-based models never look at them
    run_switch_design(make_inputs(), make_config(protonation=propka))


def test_engine_helpers_choose_the_labeller_and_the_pka_runs(engine):
    from mpnn.transforms.propka_labeller import PropkaLabeller

    assert engine._protonation_classifier(ProtonationSource()) is None
    labeller = engine._protonation_classifier(ProtonationSource("propka", 6.0, 0.25))
    assert labeller == PropkaLabeller(6.0, 0.25)
    assert engine._as_protonation(None) == ProtonationSource()
    assert engine._as_protonation({"source": "propka", "ph": 6.0}).margin == 0.5

    def needs(**overrides):
        return engine._needs_site_pka(make_config(**overrides))

    propka = {"source": "propka", "ph": 7.4}
    assert needs(protonation=propka, **LINKED)
    assert not needs(protonation=propka)  # state_binding has no use for them
    assert not needs(**LINKED)  # EV6 labels: the per-type constants
    assert isinstance(make_config(**LINKED), SwitchDesignConfig)
