from __future__ import annotations

import json
import re
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from macro_llm_tournament import data_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DynamicMacroIntegrationTests(unittest.TestCase):
    def test_installed_commands_and_make_targets_cover_dynamic_lane(self) -> None:
        pyproject = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        scripts = pyproject["project"]["scripts"]
        expected_scripts = {
            "macro-llm-prepare-dynamic-macro-panel": "macro_llm_tournament.prepare_dynamic_macro_panel:main",
            "macro-llm-frozen-vintage-bundle": "macro_llm_tournament.frozen_vintage_bundle:main",
            "macro-llm-dynamic-macro-economy": "macro_llm_tournament.dynamic_macro_economy:main",
            "macro-llm-dynamic-macro-tournament": "macro_llm_tournament.dynamic_macro_tournament:main",
            "macro-llm-data-provenance": "macro_llm_tournament.data_provenance:main",
        }
        self.assertEqual(
            {name: scripts.get(name) for name in expected_scripts},
            expected_scripts,
        )

        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
        for target in (
            "data-provenance",
            "dynamic-macro-panel",
            "dynamic-macro-bundle-fixture",
            "dynamic-macro-bundle-dev",
            "dynamic-macro-economy-fixture",
            "dynamic-macro-tournament-fixture",
            "dynamic-macro-tournament-live",
            "dynamic-macro-tournament-resume",
            "dynamic-macro-policy-tournament-fixture",
            "dynamic-macro-policy-tournament-live",
            "dynamic-macro-policy-tournament-resume",
            "dynamic-macro-policy-partial-fixture",
            "dynamic-macro-policy-partial-live",
            "dynamic-macro-policy-partial-resume",
        ):
            self.assertRegex(makefile, rf"(?m)^{re.escape(target)}:[^\n]*$")

    def test_provenance_recurses_into_dynamic_assets_and_child_runs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / "work" / "dynamic_macro" / "bundle"
            bundle.mkdir(parents=True)
            (bundle / "manifest.json").write_text(
                json.dumps({"bundle_sha256": "bundle"}), encoding="utf-8"
            )
            child = root / "outputs" / "tournament" / "candidates" / "candidate"
            child.mkdir(parents=True)
            (child / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "child_v1",
                        "status": "complete",
                        "live_call_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(data_provenance, "PROJECT_ROOT", root):
                events = data_provenance.build_events()
            by_path = {event["path"]: event for event in events}
            self.assertIn("work/dynamic_macro/bundle/manifest.json", by_path)
            child_path = "outputs/tournament/candidates/candidate/manifest.json"
            self.assertIn(child_path, by_path)
            self.assertEqual(
                by_path[child_path]["dataset"],
                "tournament/candidates/candidate",
            )


if __name__ == "__main__":
    unittest.main()
