"""
SwitchLingua 2.0 — Stage 2: Voice Profile Assigner

Maps speaker persona attributes to voice profiles for TTS synthesis.
Given a dialogue dict from Stage 1 JSONL, infers each speaker's gender and
age group from persona metadata, then selects the best-matching voice
profile from a YAML catalogue.  Speaker A and B are guaranteed to receive
different profiles.

Match priority (descending): language > gender > age_group > accent.
"""

import re
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger("voice_assigner")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VoiceProfile:
    """A single voice profile loaded from profiles.yaml."""

    id: str
    gender: str          # "male" / "female"
    age_group: str       # "young" / "middle" / "senior"
    languages: list      # e.g. ["zh", "en"]
    accent: str          # "mainland" / "hongkong" / "singapore" / "malaysia" / …
    audio_file: str      # relative path to reference audio clip


# ---------------------------------------------------------------------------
# Region → accent mapping (best-effort)
# ---------------------------------------------------------------------------

_REGION_TO_ACCENT: dict[str, str] = {
    "中国大陆": "mainland",
    "mainland": "mainland",
    "北京": "mainland",
    "上海": "mainland",
    "香港": "hongkong",
    "hongkong": "hongkong",
    "hong kong": "hongkong",
    "台湾": "taiwan",
    "taiwan": "taiwan",
    "新加坡": "singapore",
    "singapore": "singapore",
    "马来西亚": "malaysia",
    "malaysia": "malaysia",
}


def _normalise_accent(region: str) -> str:
    """Map a free-text region string to a canonical accent tag."""
    region_lower = region.lower()
    for key, accent in _REGION_TO_ACCENT.items():
        if key in region_lower:
            return accent
    return region_lower


# ---------------------------------------------------------------------------
# Main assigner
# ---------------------------------------------------------------------------

class VoiceAssigner:
    """Load voice profiles and assign them to dialogue speakers."""

    # -- construction -------------------------------------------------------

    def __init__(self, profiles_path: str | Path) -> None:
        """Load voice profiles from a YAML file.

        Parameters
        ----------
        profiles_path : str | Path
            Path to ``profiles.yaml``.  Expected schema::

                profiles:
                  - id: "voice_01"
                    gender: "male"
                    age_group: "young"
                    languages: ["zh", "en"]
                    accent: "mainland"
                    audio_file: "audio/voice_01.wav"
                  - ...
        """
        self._profiles_path = Path(profiles_path)
        self._profiles: list[VoiceProfile] = []
        self._index_by_lang: dict[str, list[VoiceProfile]] = {}

        self._load(self._profiles_path)

    def _load(self, path: Path) -> None:
        """Parse YAML and build internal indices."""
        if not path.exists():
            raise FileNotFoundError(f"Voice profiles file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        raw_profiles = data.get("profiles", [])
        if not raw_profiles:
            raise ValueError(f"No profiles found in {path}")

        for entry in raw_profiles:
            profile = VoiceProfile(
                id=str(entry["id"]),
                gender=entry.get("gender", "unknown"),
                age_group=entry.get("age_group", "middle"),
                languages=list(entry.get("languages", [])),
                accent=entry.get("accent", ""),
                audio_file=entry.get("audio_file", ""),
            )
            self._profiles.append(profile)
            for lang in profile.languages:
                self._index_by_lang.setdefault(lang, []).append(profile)

        logger.info("Loaded %d voice profiles from %s", len(self._profiles), path)

    # -- inference helpers --------------------------------------------------

    @staticmethod
    def _infer_gender(persona_desc: str) -> str:
        """Infer gender from persona description text.

        Returns ``"female"``, ``"male"``, or ``"unknown"``.
        """
        female_keywords = [
            "女", "female", "她", "mother", "sister",
            "妈", "姐", "妹", "小姐", "阿姨", "lady", "woman",
        ]
        male_keywords = [
            "男", "male", "他", "father", "brother",
            "爸", "哥", "弟", "先生", "叔", "uncle", "man",
        ]

        text = persona_desc.lower()

        female_score = sum(1 for kw in female_keywords if kw.lower() in text)
        male_score = sum(1 for kw in male_keywords if kw.lower() in text)

        if female_score > male_score:
            return "female"
        if male_score > female_score:
            return "male"
        return "unknown"

    @staticmethod
    def _infer_age_group(persona_desc: str, profession: str) -> str:
        """Infer age group from persona description and profession.

        Returns ``"young"``, ``"senior"``, or ``"middle"``.
        """
        combined = f"{persona_desc} {profession}".lower()

        young_keywords = [
            "学生", "student", "大学", "university", "研究生",
            "grad", "实习", "intern", "college", "undergraduate",
            "读研", "读博", "bachelor",
        ]
        senior_keywords = [
            "教授", "professor", "senior", "退休", "retired",
            "老", "elder", "veteran", "资深",
        ]

        if any(kw in combined for kw in young_keywords):
            return "young"
        if any(kw in combined for kw in senior_keywords):
            return "senior"
        return "middle"

    # -- matching logic -----------------------------------------------------

    def _match_profile(
        self,
        language_pair: list[str],
        gender: str,
        age_group: str,
        region: str,
        exclude_ids: set[str] | None = None,
    ) -> Optional[VoiceProfile]:
        """Find the best matching profile, excluding already-used IDs.

        Scoring (additive):
            +8  supports at least one language in the pair
            +4  gender matches (skipped when gender == "unknown")
            +2  age_group matches
            +1  accent matches

        Parameters
        ----------
        language_pair : list[str]
            e.g. ``["zh", "en"]``
        gender, age_group, region : str
            Inferred attributes of the speaker.
        exclude_ids : set[str] | None
            Profile IDs that must not be reused (guarantees different voices).

        Returns
        -------
        VoiceProfile or None
        """
        if exclude_ids is None:
            exclude_ids = set()

        target_accent = _normalise_accent(region)

        # Gather candidate profiles that support at least one required language
        lang_candidates: list[VoiceProfile] = []
        for lang in language_pair:
            lang_candidates.extend(self._index_by_lang.get(lang, []))
        # Deduplicate while preserving order
        seen: set[str] = set()
        candidates: list[VoiceProfile] = []
        for p in lang_candidates:
            if p.id not in seen and p.id not in exclude_ids:
                seen.add(p.id)
                candidates.append(p)

        # If no language-matched candidates, fall back to ALL profiles
        if not candidates:
            logger.warning(
                "No profiles matching languages %s; falling back to all profiles.",
                language_pair,
            )
            candidates = [p for p in self._profiles if p.id not in exclude_ids]

        if not candidates:
            logger.error("No available voice profiles left (all excluded).")
            return None

        # Score each candidate
        def _score(p: VoiceProfile) -> float:
            s = 0.0
            # Language: L1 (first in pair) match is critical
            l1 = language_pair[0] if language_pair else ""
            if l1 and l1 in p.languages:
                s += 16.0  # strong preference for L1 match
            elif set(p.languages) & set(language_pair):
                s += 4.0   # weaker: only L2 overlap
            # Gender
            if gender != "unknown" and p.gender == gender:
                s += 4.0
            # Age group
            if p.age_group == age_group:
                s += 2.0
            # Accent
            if target_accent and p.accent == target_accent:
                s += 1.0
            return s

        scored = [(p, _score(p)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_profile, best_score = scored[0]

        # Log warnings for imperfect matches
        max_possible = 8.0 + (4.0 if gender != "unknown" else 0.0) + 2.0 + 1.0
        if best_score < max_possible:
            missing: list[str] = []
            if not (set(best_profile.languages) & set(language_pair)):
                missing.append("language")
            if gender != "unknown" and best_profile.gender != gender:
                missing.append(f"gender(want={gender}, got={best_profile.gender})")
            if best_profile.age_group != age_group:
                missing.append(f"age(want={age_group}, got={best_profile.age_group})")
            if target_accent and best_profile.accent != target_accent:
                missing.append(f"accent(want={target_accent}, got={best_profile.accent})")
            if missing:
                logger.warning(
                    "Imperfect match for profile %s — mismatches: %s",
                    best_profile.id,
                    ", ".join(missing),
                )

        # Among ties, pick randomly to add variety
        top_score = scored[0][1]
        top_profiles = [p for p, s in scored if s == top_score]
        if len(top_profiles) > 1:
            best_profile = random.choice(top_profiles)

        return best_profile

    # -- public API ---------------------------------------------------------

    def assign_voices(self, dialogue: dict) -> tuple[VoiceProfile, VoiceProfile]:
        """Assign voice profiles to speaker A and B in a dialogue.

        Parameters
        ----------
        dialogue : dict
            One dialogue dict from Stage 1 JSONL, expected to contain
            ``speaker_a``, ``speaker_b``, and ``language_pair`` keys.

        Returns
        -------
        tuple[VoiceProfile, VoiceProfile]
            ``(voice_a, voice_b)`` — guaranteed to be different profiles.

        Raises
        ------
        RuntimeError
            If profiles cannot be assigned to both speakers.
        """
        lang_pair: list[str] = dialogue.get("language_pair", ["zh", "en"])

        # -- Speaker A -------------------------------------------------------
        sp_a: dict = dialogue.get("speaker_a", {})
        gender_a = self._infer_gender(sp_a.get("persona_description", ""))
        age_a = self._infer_age_group(
            sp_a.get("persona_description", ""),
            sp_a.get("profession", ""),
        )
        region_a = sp_a.get("region", "")

        voice_a = self._match_profile(lang_pair, gender_a, age_a, region_a)
        if voice_a is None:
            raise RuntimeError(
                f"Could not assign voice for speaker A in dialogue "
                f"{dialogue.get('dialogue_id', '?')}"
            )

        logger.info(
            "Dialogue %s — Speaker A: gender=%s age=%s region=%s → %s",
            dialogue.get("dialogue_id", "?"),
            gender_a, age_a, region_a, voice_a.id,
        )

        # -- Speaker B (must differ from A) ----------------------------------
        sp_b: dict = dialogue.get("speaker_b", {})
        gender_b = self._infer_gender(sp_b.get("persona_description", ""))
        age_b = self._infer_age_group(
            sp_b.get("persona_description", ""),
            sp_b.get("profession", ""),
        )
        region_b = sp_b.get("region", "")

        voice_b = self._match_profile(
            lang_pair, gender_b, age_b, region_b,
            exclude_ids={voice_a.id},
        )
        if voice_b is None:
            raise RuntimeError(
                f"Could not assign a *different* voice for speaker B in dialogue "
                f"{dialogue.get('dialogue_id', '?')}"
            )

        logger.info(
            "Dialogue %s — Speaker B: gender=%s age=%s region=%s → %s",
            dialogue.get("dialogue_id", "?"),
            gender_b, age_b, region_b, voice_b.id,
        )

        return voice_a, voice_b


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    """Quick smoke-test with a sample dialogue."""
    import json

    logging.basicConfig(level=logging.DEBUG)

    sample_profiles_path = Path(__file__).parent / "voice_profiles" / "profiles.yaml"

    if not sample_profiles_path.exists():
        logger.error("Demo requires %s — create it first.", sample_profiles_path)
        return

    assigner = VoiceAssigner(sample_profiles_path)

    sample_dialogue = {
        "dialogue_id": "DLG_demo",
        "language_pair": ["zh", "en"],
        "speaker_a": {
            "persona_description": "在一线城市高校工作的研究人员",
            "profession": "研究人员",
            "region": "中国大陆一线城市",
        },
        "speaker_b": {
            "persona_description": "在香港读大学的本地学生",
            "profession": "大学生",
            "region": "香港",
        },
    }

    voice_a, voice_b = assigner.assign_voices(sample_dialogue)
    print(f"Speaker A → {voice_a}")
    print(f"Speaker B → {voice_b}")


if __name__ == "__main__":
    _demo()
