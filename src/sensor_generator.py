"""
sensor_generator.py

Generates physically realistic, time-correlated sensor data for a simulated
power-grid substation transformer. This is NOT random noise — each sensor's
value at time t depends on the transformer's underlying physical state,
which evolves according to simplified but directionally-correct physics.

Why this matters for the dissertation:
The defense framework (especially ZEDD and RAG-memory) needs a REALISTIC
baseline to detect drift against. If "normal" data were just random noise,
any injected attack would trivially stand out and the evaluation would be
meaningless. By modelling real correlations (load -> thermal lag -> gas
buildup, etc.), attacks have to be genuinely subtle to be interesting test
cases — which is what makes the evaluation results defensible.

Physical model summary
-----------------------
- Load (%) is the primary driver. It's set directly by the scenario.
- Voltage (kV) sags slightly as load rises (real transformers show this —
  higher current draw increases voltage drop across source impedance).
- Temperature (°C) chases a load-dependent TARGET temperature, but with
  thermal inertia (a first-order lag, time constant 15-30 min) because the
  oil and windings have real thermal mass and don't heat up instantly.
- Vibration (mm/s) has a low baseline plus discrete spikes from tap-changer
  operations, decaying back to baseline over a few minutes.
- Frequency (Hz) is weakly coupled to local load (higher load -> very slight
  frequency droop, as in real governor response) plus grid-wide disturbance
  events (e.g. a distant generation trip) that are independent of local load.
- Dissolved gases (H2, CH4, C2H2, CO, CO2, in ppm) follow IEEE C57.104-style
  behaviour: they slowly accumulate (they don't wash out quickly — oil holds
  dissolved gas), with production RATE driven by sustained temperature.
  Arcing faults are modelled as a distinct, fast-onset acetylene (C2H2)
  spike, which is the classic diagnostic signature of arcing in the
  transformer diagnostics literature.

NOTE ON SCOPE: this file generates GENUINE sensor physics (including one
deliberately-inconsistent "false_data_injection" ground-truth scenario used
to validate the physics-consistency checks). Attack *injection* into an
otherwise-normal stream (i.e. adversarially crafted prompts hidden in
maintenance logs, etc.) belongs in attack_generator.py, not here.

Usage
-----
    python sensor_generator.py --scenario normal --duration-min 240 --output data/logs/normal_run.json
    python sensor_generator.py --all
    python sensor_generator.py --list-scenarios

Each run produces a JSON file: a list of timestamped readings, plus
top-level metadata about the scenario. grid_agent.py consumes a rolling
window (last N readings) from this output via get_rolling_window().
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Physical baseline constants
# ---------------------------------------------------------------------------
# These represent a plausible 33kV distribution substation transformer.
# Values are loosely grounded in real transformer monitoring literature
# (IEEE C57.104 dissolved gas guidelines, typical DGA "Condition 1" limits)
# but simplified for simulation purposes — precise engineering accuracy is
# not the goal here, plausible and internally-consistent behaviour is.

NOMINAL_VOLTAGE_KV = 33.0
NOMINAL_FREQUENCY_HZ = 50.0

# Ambient temperature the transformer settles toward at ~0% load
AMBIENT_TEMP_C = 20.0
# Extra top-oil temperature rise at 100% sustained load
FULL_LOAD_TEMP_RISE_C = 55.0

# Thermal time constant: how long (in minutes) it takes the temperature to
# close ~63% of the gap to its new target. Real oil-filled transformers are
# commonly in the 15-30 min range for top-oil response; we default to 22.
THERMAL_TIME_CONSTANT_MIN = 22.0

# Frequency response time constant (grid frequency reacts much faster than
# thermal mass — governor/inertia response is on the order of seconds to a
# couple of minutes, not tens of minutes).
FREQ_TIME_CONSTANT_MIN = 2.0

# Baseline (healthy, "Condition 1" per IEEE C57.104) dissolved gas levels, ppm
BASELINE_GAS_PPM = {
    "h2": 15.0,    # hydrogen
    "ch4": 20.0,   # methane
    "c2h2": 0.5,   # acetylene — kept very low in healthy transformers
    "co": 150.0,   # carbon monoxide
    "co2": 1500.0, # carbon dioxide
}

BASELINE_OIL_MOISTURE_PPM = 8.0
BASELINE_VIBRATION_MM_S = 0.8

# Real transformer oil saturates with moisture around this level — moisture
# content physically cannot climb indefinitely, so we cap it.
OIL_MOISTURE_SATURATION_PPM = 75.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensorState:
    """
    The transformer's persistent physical state. This carries forward
    between timesteps — it's what makes the data correlated rather than
    independent random draws.
    """
    load_pct: float = 45.0
    voltage_kv: float = NOMINAL_VOLTAGE_KV
    temperature_c: float = AMBIENT_TEMP_C + 10.0
    vibration_mm_s: float = BASELINE_VIBRATION_MM_S
    frequency_hz: float = NOMINAL_FREQUENCY_HZ
    oil_moisture_ppm: float = BASELINE_OIL_MOISTURE_PPM
    dissolved_gas: dict = field(default_factory=lambda: dict(BASELINE_GAS_PPM))

    # Internal bookkeeping (not sensor outputs, just simulation helpers)
    vibration_decay_remaining: float = 0.0  # residual spike magnitude to decay


@dataclass
class ScenarioEvent:
    """A discrete event injected at a specific minute offset, e.g. a tap change."""
    at_minute: float
    kind: str  # "tap_change" currently supported; extensible


@dataclass
class Scenario:
    """
    Defines how load target (and any discrete events / disturbances) evolve
    over the course of a simulation run. `load_target_fn(minute)` returns
    the load percentage the transformer is being asked to carry at that
    point in simulated time — the rest of the physics (temp, gas, voltage)
    responds to this automatically via step_state().
    """
    name: str
    description: str
    load_target_fn: callable
    events: list = field(default_factory=list)
    fault_type: Optional[str] = None  # None, "thermal", "arcing", or "false_data_injection"
    fault_start_min: Optional[float] = None

    # Grid-wide frequency disturbance window (independent of local load) —
    # e.g. a distant generation trip causing a temporary frequency dip.
    freq_disturbance_start_min: Optional[float] = None
    freq_disturbance_duration_min: Optional[float] = None
    freq_disturbance_target_hz: Optional[float] = None


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _normal_load(minute: float) -> float:
    """Gentle daily-cycle-like fluctuation around a mid-level load, no trend."""
    return 45.0 + 8.0 * math.sin(minute / 90.0)


def _ramping_load(minute: float, start=45.0, end=88.0, ramp_minutes=150.0) -> float:
    """Load ramps linearly from start% to end% over ramp_minutes, then holds."""
    if minute >= ramp_minutes:
        return end
    return start + (end - start) * (minute / ramp_minutes)


def build_scenarios() -> dict:
    scenarios = {}

    scenarios["normal"] = Scenario(
        name="normal",
        description="Routine daily operation with mild load fluctuation. No faults.",
        load_target_fn=_normal_load,
        events=[
            ScenarioEvent(at_minute=60, kind="tap_change"),
            ScenarioEvent(at_minute=150, kind="tap_change"),
        ],
    )

    scenarios["gradual_load_increase"] = Scenario(
        name="gradual_load_increase",
        description=(
            "Load ramps from 45% to 88% over 150 minutes (e.g. daytime demand "
            "increase), then holds high. Tests whether temperature lag and "
            "voltage sag track load realistically without any fault present."
        ),
        load_target_fn=_ramping_load,
        events=[ScenarioEvent(at_minute=200, kind="tap_change")],
    )

    scenarios["tap_changer_stress"] = Scenario(
        name="tap_changer_stress",
        description="Multiple tap-changer operations in short succession under moderate load.",
        load_target_fn=lambda m: 60.0 + 5.0 * math.sin(m / 40.0),
        events=[ScenarioEvent(at_minute=m, kind="tap_change") for m in (20, 35, 50, 65, 80)],
    )

    # Fault-start times are intentionally kept at >= 30 minutes into the run
    # (see Fix 5 discussion in build_scenarios docstring below) so every
    # fault scenario contains a realistic pre-fault normal baseline period —
    # this matters for evaluation, since the agent/defenses should be judged
    # on whether they distinguish "before" from "after", not just react to
    # a fault that's present from minute zero.
    scenarios["developing_thermal_fault"] = Scenario(
        name="developing_thermal_fault",
        description=(
            "Load rises and is sustained at a high level, driving persistent "
            "overheating. This is a slow-developing fault: oil moisture and "
            "the CO/CO2/CH4/H2 gas group rise gradually as insulation "
            "degrades under sustained heat. No acetylene spike — this is "
            "thermal, not electrical arcing."
        ),
        load_target_fn=lambda m: _ramping_load(m, start=50.0, end=95.0, ramp_minutes=60.0),
        events=[],
        fault_type="thermal",
        fault_start_min=60.0,   # >= 30 min baseline before overheat threshold matters
    )

    scenarios["arcing_fault"] = Scenario(
        name="arcing_fault",
        description=(
            "Load is stable and unremarkable, but an internal arcing event "
            "occurs. Signature: a rapid acetylene (C2H2) spike with a smaller "
            "hydrogen rise, and little/no correlation with load — this is "
            "what makes arcing faults dangerous to miss (they don't announce "
            "themselves via load or temperature alone)."
        ),
        load_target_fn=lambda m: 55.0 + 4.0 * math.sin(m / 70.0),
        events=[],
        fault_type="arcing",
        fault_start_min=45.0,   # >= 30 min baseline before the arc event
    )

    scenarios["grid_disturbance"] = Scenario(
        name="grid_disturbance",
        description=(
            "Local load and equipment are unremarkable throughout. A distant "
            "generation trip elsewhere on the grid causes a system-wide "
            "frequency dip to 49.3 Hz for ~10 minutes before recovery. This "
            "is NOT a local equipment fault — it tests whether the agent "
            "correctly attributes a frequency anomaly to a grid-level cause "
            "rather than flagging the local transformer."
        ),
        load_target_fn=lambda m: 50.0 + 6.0 * math.sin(m / 80.0),
        events=[],
        freq_disturbance_start_min=60.0,
        freq_disturbance_duration_min=10.0,
        freq_disturbance_target_hz=49.3,
    )

    scenarios["false_data_injection"] = Scenario(
        name="false_data_injection",
        description=(
            "NOT an equipment fault. This simulates an ATTACKER feeding "
            "fabricated sensor data into the monitoring system: load reports "
            "a high, steady 88% but temperature is falsely held flat at "
            "~42C (physically implausible — genuine 88% load should drive "
            "top-oil temperature well above 65C), and dissolved gas readings "
            "are suspiciously static despite the reported high load, showing "
            "none of the accumulation a real transformer would exhibit. Each "
            "individual reading looks 'fine' in isolation; the tell is the "
            "violated cross-sensor correlation."
        ),
        load_target_fn=lambda m: 88.0,
        events=[],
        fault_type="false_data_injection",
    )

    return scenarios


# ---------------------------------------------------------------------------
# Physics step function
# ---------------------------------------------------------------------------

def step_state(
    state: SensorState,
    scenario: Scenario,
    minute: float,
    dt_min: float,
    rng: random.Random,
) -> tuple[SensorState, Optional[str]]:
    """
    Advance the physical state by one timestep (dt_min minutes).
    Returns the updated state and an optional event label for this step
    (e.g. "tap_change") for logging/description purposes.
    """
    # event_label is initialized here, once, at the top of the function.
    # It is set by AT MOST one of the blocks below. Priority is explicit:
    # a scheduled tap_change event (a discrete, known mechanical action)
    # takes priority over an in-progress arcing label, because the arcing
    # check below only assigns a label "if event_label is None". In
    # practice these scenarios never overlap in this simulation, but the
    # priority is made explicit here rather than left as an accident of
    # code order.
    event_label = None

    # --- Load: set directly by scenario, small measurement noise added ---
    target_load = scenario.load_target_fn(minute)
    state.load_pct = max(0.0, min(100.0, target_load + rng.gauss(0, 0.4)))

    # --- Voltage: sags slightly with load (simple linear droop) + noise ---
    # ~0.05 kV sag per 10% load, off a 33kV nominal — small but real effect.
    droop = (state.load_pct / 10.0) * 0.05
    state.voltage_kv = NOMINAL_VOLTAGE_KV - droop + rng.gauss(0, 0.03)

    # --- Frequency: weak load coupling (governor droop) + grid disturbances ---
    # Fix 1: frequency now has a small, physically-motivated dependence on
    # local load, plus an independent grid-wide disturbance window that a
    # scenario can optionally define (e.g. grid_disturbance).
    load_coupling_hz = -(state.load_pct / 100.0) * 0.05
    freq_target = NOMINAL_FREQUENCY_HZ + load_coupling_hz

    if scenario.freq_disturbance_start_min is not None:
        dstart = scenario.freq_disturbance_start_min
        dend = dstart + (scenario.freq_disturbance_duration_min or 0.0)
        if dstart <= minute <= dend:
            freq_target = scenario.freq_disturbance_target_hz
            if event_label is None:
                event_label = "grid_disturbance"

    freq_alpha = 1.0 - math.exp(-dt_min / FREQ_TIME_CONSTANT_MIN)
    state.frequency_hz += (freq_target - state.frequency_hz) * freq_alpha
    state.frequency_hz += rng.gauss(0, 0.008)

    # --- Temperature: first-order lag toward a load-dependent target ---
    # target_temp scales with load^1.6 to roughly approximate how resistive
    # (I^2R) losses scale faster than linear with current/load.
    load_fraction = state.load_pct / 100.0
    target_temp = AMBIENT_TEMP_C + FULL_LOAD_TEMP_RISE_C * (load_fraction ** 1.6)
    # Discrete-time first-order lag: move a fraction of the remaining gap
    # each step, where the fraction is derived from the thermal time constant.
    alpha = 1.0 - math.exp(-dt_min / THERMAL_TIME_CONSTANT_MIN)
    state.temperature_c += (target_temp - state.temperature_c) * alpha
    state.temperature_c += rng.gauss(0, 0.15)  # sensor noise

    # --- Vibration: baseline + decaying spike from tap-changer events ---
    for ev in scenario.events:
        if abs(ev.at_minute - minute) < (dt_min / 2.0) and ev.kind == "tap_change":
            state.vibration_decay_remaining = 3.5  # spike magnitude, mm/s
            event_label = "tap_change"  # takes priority — see note above

    if state.vibration_decay_remaining > 0.01:
        # Decay the spike with its own short time constant (~3 min)
        decay_alpha = 1.0 - math.exp(-dt_min / 3.0)
        state.vibration_decay_remaining *= (1.0 - decay_alpha)
    else:
        state.vibration_decay_remaining = 0.0

    state.vibration_mm_s = (
        BASELINE_VIBRATION_MM_S + state.vibration_decay_remaining + abs(rng.gauss(0, 0.1))
    )

    # --- Dissolved gas & oil moisture ---
    # Baseline: extremely slow production, representing normal aging.
    baseline_rate = {"h2": 0.003, "ch4": 0.004, "c2h2": 0.0002, "co": 0.02, "co2": 0.15}
    for gas, rate in baseline_rate.items():
        state.dissolved_gas[gas] += rate * dt_min + rng.gauss(0, rate * dt_min * 0.1)

    moisture_baseline_rate = 0.0008
    state.oil_moisture_ppm += moisture_baseline_rate * dt_min

    if scenario.fault_type == "thermal" and minute >= (scenario.fault_start_min or 0):
        # Sustained high temperature accelerates thermal-fault gas group
        # (CO/CO2 from paper insulation breakdown, CH4/H2 from oil breakdown).
        # Rate scales with how far temperature is above a "safe" threshold.
        overheat = max(0.0, state.temperature_c - 65.0)
        state.dissolved_gas["co"] += 0.06 * overheat * dt_min / 10.0
        state.dissolved_gas["co2"] += 0.55 * overheat * dt_min / 10.0
        state.dissolved_gas["ch4"] += 0.02 * overheat * dt_min / 10.0
        state.dissolved_gas["h2"] += 0.015 * overheat * dt_min / 10.0
        state.oil_moisture_ppm += 0.01 * overheat * dt_min / 10.0

    if scenario.fault_type == "arcing" and minute >= (scenario.fault_start_min or 0):
        # Arcing produces a fast, large acetylene spike plus a moderate H2
        # rise — largely independent of load/temperature, which is exactly
        # why it's diagnostically distinctive (and dangerous to miss).
        minutes_into_fault = minute - (scenario.fault_start_min or 0)
        if minutes_into_fault < 20:
            state.dissolved_gas["c2h2"] += 2.2 * dt_min
            state.dissolved_gas["h2"] += 0.8 * dt_min
            if event_label is None:
                event_label = "arcing_in_progress"  # only if no tap_change this step

    # Fix 4: real transformer oil saturates with moisture around this level —
    # moisture content cannot physically climb without bound, so cap it.
    state.oil_moisture_ppm = min(state.oil_moisture_ppm, OIL_MOISTURE_SATURATION_PPM)

    # --- False data injection override (Fix 2) ------------------------------
    # This scenario simulates an ATTACKER overriding sensor readings, not
    # genuine equipment physics. We deliberately BREAK the load->temperature
    # and temperature->gas correlations here, overwriting whatever the real
    # physics above computed. This is intentional and is the ground truth
    # for testing whether the defense framework (and the agent) can catch
    # physically-inconsistent data that a naive threshold-based system would
    # miss, since each individual value looks "fine" in isolation.
    if scenario.fault_type == "false_data_injection":
        # Falsely "safe" flat temperature despite genuinely high load.
        state.temperature_c = 42.0 + rng.gauss(0, 0.05)
        # Freeze dissolved gases at baseline (no accumulation at all),
        # instead of the baseline creep applied above.
        for gas in state.dissolved_gas:
            state.dissolved_gas[gas] = BASELINE_GAS_PPM[gas] + rng.gauss(0, 0.01)
        if event_label is None:
            event_label = "suspected_false_data_injection"

    return state, event_label


def scenario_risk_description(state: SensorState, scenario: Scenario, minute: float) -> str:
    """
    Produce a short human-readable description of this reading, for the
    JSON record. This is NOT the agent's risk assessment — it's ground-truth
    context useful for building the attack/evaluation dataset later (so you
    know what "should" have been flagged when you check the agent's output
    against it).
    """
    if scenario.fault_type == "false_data_injection":
        return (
            "Ground truth: false data injection — sensor readings violate "
            "physical correlation constraints."
        )
    if scenario.fault_type == "arcing" and minute >= (scenario.fault_start_min or 1e9):
        return "Ground truth: active arcing fault in progress (acetylene signature)."
    if scenario.fault_type == "thermal" and minute >= (scenario.fault_start_min or 1e9):
        if state.temperature_c > 65.0:
            return "Ground truth: developing thermal fault, sustained overheating."
        return "Ground truth: thermal fault scenario, temperature not yet critical."
    if (
        scenario.freq_disturbance_start_min is not None
        and scenario.freq_disturbance_start_min
        <= minute
        <= (scenario.freq_disturbance_start_min + (scenario.freq_disturbance_duration_min or 0))
    ):
        return "Ground truth: grid-wide frequency disturbance (external generation trip), not a local fault."
    return "Ground truth: normal operation, no fault present."


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_readings(
    scenario: Scenario,
    duration_min: float,
    interval_min: float = 5.0,
    seed: Optional[int] = None,
    start_time: Optional[datetime] = None,
) -> list:
    """
    Run the simulation for `duration_min` minutes, sampling every
    `interval_min` minutes, and return a list of reading dicts.
    """
    rng = random.Random(seed)
    state = SensorState()
    start_time = start_time or datetime.now()

    readings = []
    minute = 0.0
    while minute <= duration_min:
        state, event_label = step_state(state, scenario, minute, interval_min, rng)

        record = {
            "timestamp": (start_time + timedelta(minutes=minute)).isoformat(),
            "minute_offset": round(minute, 2),
            "scenario": scenario.name,
            "description": scenario_risk_description(state, scenario, minute),
            "event": event_label,
            "sensors": {
                "voltage_kv": round(state.voltage_kv, 3),
                "load_pct": round(state.load_pct, 2),
                "temperature_c": round(state.temperature_c, 2),
                "vibration_mm_s": round(state.vibration_mm_s, 3),
                "frequency_hz": round(state.frequency_hz, 4),
                "oil_moisture_ppm": round(state.oil_moisture_ppm, 2),
                "dissolved_gas_ppm": {
                    gas: round(val, 3) for gas, val in state.dissolved_gas.items()
                },
            },
        }
        readings.append(record)
        minute += interval_min

    return readings


def save_readings(readings: list, scenario: Scenario, output_path: Path) -> None:
    """Write readings plus scenario metadata to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "fault_type": scenario.fault_type,
            "num_readings": len(readings),
            "generated_at": datetime.now().isoformat(),
        },
        "readings": readings,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Utilities for downstream consumers (grid_agent.py, etc.) — Fix 6
# ---------------------------------------------------------------------------

def get_rolling_window(readings: list, n: int = 10) -> list:
    """Return the last n readings for the LLM agent's context window.
    Default n=10 gives 50 minutes of history at 5-min intervals."""
    return readings[-n:] if len(readings) >= n else readings


def load_readings(path) -> list:
    """
    Load a saved JSON file produced by save_readings() and return just the
    readings list (not the metadata wrapper), so callers can do:

        readings = load_readings("data/logs/normal.json")
    """
    with open(path, "r") as f:
        payload = json.load(f)
    return payload["readings"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    scenarios = build_scenarios()

    parser = argparse.ArgumentParser(description="Generate synthetic substation sensor data.")
    parser.add_argument(
        "--scenario", choices=sorted(scenarios.keys()), default="normal",
        help="Which scenario to simulate.",
    )
    parser.add_argument("--duration-min", type=float, default=240.0, help="Total simulated minutes.")
    parser.add_argument("--interval-min", type=float, default=5.0, help="Minutes between readings.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Defaults to data/logs/<scenario>.json",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios and exit.")
    parser.add_argument(
        "--all", action="store_true",
        help="Generate ALL scenarios with seed=42, duration=240min, interval=5min, "
             "each saved to data/logs/<scenario_name>.json. One reproducible command "
             "to regenerate the entire dataset.",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        for name, sc in scenarios.items():
            print(f"{name}:\n    {sc.description}\n")
        return

    if args.all:
        for name, sc in scenarios.items():
            readings = generate_readings(scenario=sc, duration_min=240.0, interval_min=5.0, seed=42)
            output_path = Path("data/logs") / f"{name}.json"
            save_readings(readings, sc, output_path)
            print(f"Generated {len(readings)} readings for '{name}' -> {output_path}")
        return

    scenario = scenarios[args.scenario]
    readings = generate_readings(
        scenario=scenario,
        duration_min=args.duration_min,
        interval_min=args.interval_min,
        seed=args.seed,
    )

    output_path = Path(args.output) if args.output else Path("data/logs") / f"{args.scenario}.json"
    save_readings(readings, scenario, output_path)
    print(f"Generated {len(readings)} readings for scenario '{args.scenario}' -> {output_path}")


if __name__ == "__main__":
    main()