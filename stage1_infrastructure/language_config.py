"""
SwitchLingua 2.0 — Language Pair Configuration
Core configuration dataclass and loader for multi-language support.

All downstream modules use LanguagePairConfig instead of hardcoded values
to get language-specific settings (prompts, personas, calibration, etc.).

Usage:
    from language_config import LanguagePairConfig

    cfg = LanguagePairConfig.load("cmn_eng")
    print(cfg.l1_name, cfg.l2_name)

    available = LanguagePairConfig.list_available()
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Project root: two levels up from this file's directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LanguagePairConfig:
    """
    Complete configuration for one language pair.

    Loaded from a config folder at ``configs/{pair_id}/`` which contains:
      - language.yaml   — identity, script, detection, punctuation, fillers
      - prompts.yaml    — role template, CS behavior templates, proficiency
                          descriptions, mixing level descriptions, generation
                          requirements, example dialogues
      - personas.yaml   — list of persona dicts
      - calibration.json — corpus calibration statistics
      - evaluation.yaml  — rule-based evaluator settings
      - l1_lexicon.txt   (optional) — one word per line, used for lexicon
                          based language detection
    """

    # ---- Identity ----
    pair_id: str
    l1_code: str
    l2_code: str
    l1_name: str
    l2_name: str
    l1_name_en: str
    l2_name_en: str

    # ---- Script / detection ----
    l1_script: str
    l2_script: str
    detection_mode: str  # "unicode" or "lexicon"
    l1_unicode_ranges: list = field(default_factory=list)
    l2_unicode_ranges: list = field(default_factory=list)

    # ---- Punctuation ----
    sentence_endings: list = field(default_factory=list)
    l1_punctuation: list = field(default_factory=list)

    # ---- Filler words ----
    filler_words_l1: list = field(default_factory=list)
    filler_words_l2: list = field(default_factory=list)
    interjections_l1: list = field(default_factory=list)

    # ---- Prompt templates ----
    role_template: str = ""
    cs_behavior_templates: dict = field(default_factory=dict)
    proficiency_descriptions: dict = field(default_factory=dict)
    mixing_level_descriptions: dict = field(default_factory=dict)
    generation_requirements: str = ""
    example_dialogues: list = field(default_factory=list)

    # ---- Data ----
    personas: list = field(default_factory=list)
    calibration: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)

    # ---- Optional ----
    l1_lexicon: Optional[set] = field(default=None, repr=False)
    max_retries_override: Optional[int] = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load a YAML file and return its contents as a dict."""
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load a JSON file and return its contents as a dict."""
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def list_available(cls, configs_root: Optional[str] = None) -> list[str]:
        """Return a sorted list of available pair IDs.

        Each subdirectory of *configs_root* that contains a ``language.yaml``
        file is considered an available language pair.
        """
        root = Path(configs_root) if configs_root else _PROJECT_ROOT / "configs"
        if not root.is_dir():
            return []
        return sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and (d / "language.yaml").exists()
        )

    @classmethod
    def load(
        cls,
        pair_id: str,
        configs_root: Optional[str] = None,
    ) -> "LanguagePairConfig":
        """Load a full language-pair configuration from disk.

        Parameters
        ----------
        pair_id : str
            Identifier such as ``"cmn_eng"`` or ``"fra_eng"``.
        configs_root : str | None
            Override the default configs directory
            (``{project_root}/configs``).

        Returns
        -------
        LanguagePairConfig

        Raises
        ------
        FileNotFoundError
            If the requested *pair_id* directory does not exist.  The error
            message lists all available pair IDs for convenience.
        """
        root = Path(configs_root) if configs_root else _PROJECT_ROOT / "configs"
        pair_dir = root / pair_id

        if not pair_dir.is_dir():
            available = cls.list_available(str(root))
            avail_str = ", ".join(available) if available else "(none)"
            raise FileNotFoundError(
                f"Language pair '{pair_id}' not found at {pair_dir}. "
                f"Available pairs: {avail_str}"
            )

        # -- Load required files ------------------------------------------
        lang = cls._load_yaml(pair_dir / "language.yaml")
        prompts = cls._load_yaml(pair_dir / "prompts.yaml")
        personas_data = cls._load_yaml(pair_dir / "personas.yaml")
        calibration = cls._load_json(pair_dir / "calibration.json")
        evaluation = cls._load_yaml(pair_dir / "evaluation.yaml")

        # -- Optional lexicon ---------------------------------------------
        lexicon_path = pair_dir / "l1_lexicon.txt"
        l1_lexicon: Optional[set] = None
        if lexicon_path.exists():
            with open(lexicon_path, "r", encoding="utf-8") as fh:
                l1_lexicon = {
                    line.strip() for line in fh if line.strip()
                }

        # -- Build personas list ------------------------------------------
        if isinstance(personas_data, dict):
            personas_list = personas_data.get(
                "persona_templates",
                personas_data.get("personas", []),
            )
        elif isinstance(personas_data, list):
            personas_list = personas_data
        else:
            personas_list = []

        # -- Construct dataclass ------------------------------------------
        # language.yaml and prompts.yaml use flat key structure
        return cls(
            # Identity
            pair_id=pair_id,
            l1_code=lang.get("l1_code", ""),
            l2_code=lang.get("l2_code", ""),
            l1_name=lang.get("l1_name", ""),
            l2_name=lang.get("l2_name", ""),
            l1_name_en=lang.get("l1_name_en", ""),
            l2_name_en=lang.get("l2_name_en", ""),
            # Script / detection
            l1_script=lang.get("l1_script", ""),
            l2_script=lang.get("l2_script", ""),
            detection_mode=lang.get("detection_mode", "unicode"),
            l1_unicode_ranges=lang.get("l1_unicode_ranges", []),
            l2_unicode_ranges=lang.get("l2_unicode_ranges", []),
            # Punctuation
            sentence_endings=lang.get("sentence_endings", "。！？.!?"),
            l1_punctuation=lang.get("l1_punctuation", ""),
            # Fillers
            filler_words_l1=lang.get("filler_words_l1", []),
            filler_words_l2=lang.get("filler_words_l2", []),
            interjections_l1=lang.get("interjections_l1", []),
            # Prompt templates (flat keys in prompts.yaml)
            role_template=prompts.get("role_template", ""),
            cs_behavior_templates=prompts.get(
                "cs_behavior_templates", {}
            ),
            proficiency_descriptions=prompts.get(
                "proficiency_descriptions", {}
            ),
            mixing_level_descriptions=prompts.get(
                "mixing_level_descriptions", {}
            ),
            generation_requirements=prompts.get(
                "generation_requirements", ""
            ),
            example_dialogues=prompts.get("example_dialogues", []),
            # Data
            personas=personas_list,
            calibration=calibration,
            evaluation=evaluation,
            # Optional
            l1_lexicon=l1_lexicon,
            max_retries_override=lang.get("max_retries_override"),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def display_name(self) -> str:
        """Human-readable name like 'Mandarin-English'."""
        return f"{self.l1_name_en}-{self.l2_name_en}"

    @property
    def uses_lexicon_detection(self) -> bool:
        """Whether this pair requires lexicon-based language detection."""
        return self.detection_mode == "lexicon"

    def __repr__(self) -> str:
        return (
            f"LanguagePairConfig(pair_id={self.pair_id!r}, "
            f"display_name={self.display_name!r})"
        )
