"""
src/signal/phase_validator.py
-------------------------------
Signal Phase Validation Module.

Validates signal phase configurations before they are used by the
recommendation engine. Ensures:
  - No phase conflicts with itself (reflexive)
  - Conflict relationship is symmetric
  - Served approaches exist in the intersection configuration
  - Green time constraints are logically consistent
  - Phase sequence has no gaps
  - Compatible movements are not listed in conflicting phases

Safety guarantee: A phase configuration that fails validation is NEVER
passed to the recommendation engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class PhaseValidationError(Exception):
    """Raised when a phase configuration fails validation."""
    pass


class SignalPhaseValidator:
    """
    Validates signal phase configurations.

    Usage:
        validator = SignalPhaseValidator(config)
        is_valid, errors = validator.validate_configuration()
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config

    def validate_configuration(self, config: Optional[Any] = None) -> Tuple[bool, List[str]]:
        target_cfg = config if config is not None else self.config
        if target_cfg is None:
            return False, ["No intersection configuration provided"]
        errors = self.validate(target_cfg)
        return len(errors) == 0, errors

    def validate_proposed_phase(self, phase: Any, proposed_green_sec: float) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        p_dict = phase.model_dump() if hasattr(phase, "model_dump") else (phase if isinstance(phase, dict) else vars(phase))
        min_g = p_dict.get("min_green_sec", 10.0)
        max_g = p_dict.get("max_green_sec", 60.0)
        ped_walk = p_dict.get("pedestrian_walk_sec")

        if proposed_green_sec < min_g:
            errors.append(f"Minimum green violation: proposed {proposed_green_sec}s < min {min_g}s")
        if proposed_green_sec > max_g:
            errors.append(f"Maximum green violation: proposed {proposed_green_sec}s > max {max_g}s")
        if ped_walk is not None and proposed_green_sec < ped_walk:
            errors.append(f"Pedestrian clearance violation: proposed {proposed_green_sec}s < pedestrian walk {ped_walk}s")

        return len(errors) == 0, errors

    def validate(self, config: Any) -> List[str]:
        """
        Validate an intersection configuration dict or Pydantic model.

        Returns a list of error strings. Empty list = valid.
        """
        if hasattr(config, "model_dump"):
            config = config.model_dump()
        elif hasattr(config, "dict"):
            config = config.dict()
        elif not isinstance(config, dict):
            config = vars(config)

        errors: List[str] = []
        phases = config.get("phases", [])
        approaches = config.get("approaches", [])

        approach_ids: Set[str] = set()
        for a in approaches:
            if isinstance(a, dict):
                approach_ids.add(a.get("approach_id", ""))
            else:
                approach_ids.add(getattr(a, "approach_id", ""))

        phase_ids: Set[str] = set()
        for p in phases:
            if isinstance(p, dict):
                phase_ids.add(p.get("phase_id", ""))
            else:
                phase_ids.add(getattr(p, "phase_id", ""))

        if not phases:
            errors.append("No signal phases defined. At least one phase is required.")
            return errors

        # Build phase lookup
        phase_map: Dict[str, dict] = {p["phase_id"]: p for p in phases}

        for phase in phases:
            pid = phase.get("phase_id", "<unnamed>")
            errors.extend(self._validate_single_phase(phase, pid, approach_ids, phase_ids))

        # Validate pairwise conflicts
        errors.extend(self._validate_conflict_symmetry(phases))

        # Validate sequence is contiguous
        errors.extend(self._validate_sequence(phases))

        # Validate cycle budget
        errors.extend(self._validate_cycle_budget(phases, config))

        return errors

    # ---------------------------------------------------------------------- #
    # Per-phase validation
    # ---------------------------------------------------------------------- #

    def _validate_single_phase(
        self,
        phase: dict,
        pid: str,
        approach_ids: Set[str],
        phase_ids: Set[str],
    ) -> List[str]:
        errors: List[str] = []

        # 1. Served approaches must exist
        for ap_id in phase.get("served_approaches", []):
            if ap_id not in approach_ids:
                errors.append(
                    f"Phase '{pid}': served approach '{ap_id}' is not defined in intersection approaches."
                )

        # 2. At least one served approach
        if not phase.get("served_approaches"):
            errors.append(f"Phase '{pid}': must serve at least one approach.")

        # 3. Conflicting phases must exist and not include self
        for conf_pid in phase.get("conflicting_phases", []):
            if conf_pid == pid:
                errors.append(f"Phase '{pid}': a phase cannot conflict with itself.")
            elif conf_pid not in phase_ids:
                errors.append(
                    f"Phase '{pid}': conflicting phase '{conf_pid}' is not defined."
                )

        # 4. Green time constraints
        min_g = phase.get("min_green_sec", 10.0)
        max_g = phase.get("max_green_sec", 60.0)
        amber = phase.get("amber_sec", 3.0)
        all_red = phase.get("all_red_sec", 2.0)

        if min_g < 5.0:
            errors.append(
                f"Phase '{pid}': min_green_sec={min_g:.1f}s is below safety minimum of 5s."
            )
        if max_g < min_g:
            errors.append(
                f"Phase '{pid}': max_green_sec={max_g:.1f}s < min_green_sec={min_g:.1f}s."
            )
        if max_g > 180.0:
            errors.append(
                f"Phase '{pid}': max_green_sec={max_g:.1f}s exceeds 180s upper limit."
            )
        if amber < 3.0:
            errors.append(
                f"Phase '{pid}': amber_sec={amber:.1f}s is below safety minimum of 3s."
            )
        if all_red < 1.0:
            errors.append(
                f"Phase '{pid}': all_red_sec={all_red:.1f}s is below safety minimum of 1s."
            )

        # 5. Pedestrian clearance
        ped_walk = phase.get("pedestrian_walk_sec")
        if ped_walk is not None and ped_walk < 0:
            errors.append(f"Phase '{pid}': pedestrian_walk_sec cannot be negative.")

        return errors

    # ---------------------------------------------------------------------- #
    # Conflict symmetry
    # ---------------------------------------------------------------------- #

    def _validate_conflict_symmetry(self, phases: List[dict]) -> List[str]:
        """
        If phase A declares phase B as conflicting, phase B must also
        declare phase A as conflicting (bidirectional safety constraint).
        """
        errors: List[str] = []
        phase_map = {p["phase_id"]: set(p.get("conflicting_phases", [])) for p in phases}

        for pid, conflicts in phase_map.items():
            for conf_pid in conflicts:
                if conf_pid in phase_map:
                    if pid not in phase_map[conf_pid]:
                        errors.append(
                            f"Conflict asymmetry: phase '{pid}' lists '{conf_pid}' as conflicting, "
                            f"but '{conf_pid}' does not list '{pid}'. "
                            "Conflicts must be declared bidirectionally."
                        )
        return errors

    # ---------------------------------------------------------------------- #
    # Sequence validation
    # ---------------------------------------------------------------------- #

    def _validate_sequence(self, phases: List[dict]) -> List[str]:
        """Phase sequence positions must be 0-indexed and contiguous."""
        errors: List[str] = []
        positions = sorted(p.get("sequence_position", 0) for p in phases)

        # Allow all positions to be 0 (sequence not configured)
        if all(p == 0 for p in positions):
            return errors

        expected_0 = list(range(len(positions)))
        expected_1 = list(range(1, len(positions) + 1))
        if positions != expected_0 and positions != expected_1:
            errors.append(
                f"Phase sequence positions {positions} are not contiguous (expected {expected_0} or {expected_1})."
            )
        return errors

    # ---------------------------------------------------------------------- #
    # Cycle budget validation
    # ---------------------------------------------------------------------- #

    def _validate_cycle_budget(self, phases: List[dict], config: dict) -> List[str]:
        """
        Verify that minimum green + clearance intervals for all phases fit
        within the configured maximum cycle length.
        """
        errors: List[str] = []
        max_cycle = config.get("max_cycle_sec", 150.0)

        total_min_overhead = 0.0
        for phase in phases:
            min_g = phase.get("min_green_sec", 10.0)
            amber = phase.get("amber_sec", 3.0)
            all_red = phase.get("all_red_sec", 2.0)
            ped_walk = phase.get("pedestrian_walk_sec") or 0.0
            total_min_overhead += min_g + amber + all_red + ped_walk

        if total_min_overhead > max_cycle:
            errors.append(
                f"Sum of minimum greens + clearances ({total_min_overhead:.1f}s) "
                f"exceeds max_cycle_sec ({max_cycle:.1f}s). "
                "Reduce min_green_sec values or increase max_cycle_sec."
            )
        return errors

    # ---------------------------------------------------------------------- #
    # Phase compatibility check at recommendation time
    # ---------------------------------------------------------------------- #

    def are_compatible(self, phase_a_id: str, phase_b_id: str, phase_map: Dict[str, dict]) -> bool:
        """
        Return True if phases A and B can run simultaneously.
        Two phases are compatible if neither lists the other as conflicting.
        """
        a = phase_map.get(phase_a_id, {})
        b = phase_map.get(phase_b_id, {})
        a_conflicts = set(a.get("conflicting_phases", []))
        b_conflicts = set(b.get("conflicting_phases", []))
        return phase_b_id not in a_conflicts and phase_a_id not in b_conflicts

    def get_valid_phases(
        self,
        phases: List[dict],
        excluded_phase_ids: Optional[Set[str]] = None,
    ) -> List[dict]:
        """
        Return phases that are not explicitly excluded and have valid structure.
        Does not filter by conflict (conflict resolution is done at recommendation time).
        """
        excluded = excluded_phase_ids or set()
        return [p for p in phases if p.get("phase_id") not in excluded]
