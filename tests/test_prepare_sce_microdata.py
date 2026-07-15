from __future__ import annotations

import unittest

import pandas as pd

from macro_llm_tournament.prepare_sce_microdata import normalize_sce_raw_frame


class PrepareSceMicrodataTests(unittest.TestCase):
    def test_time_varying_job_loss_is_not_backfilled_from_later_wave(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "date": 202503,
                    "userid": 1,
                    "weight": 1.0,
                    "Q4new": 40.0,
                    "Q9_cent50": 3.0,
                    "Q25v2part2": 4.0,
                    "Q13new": pd.NA,
                }
            ]
        )
        lookup = pd.DataFrame(
            [{"userid": 1, "Q13new": 80.0, "_HH_INC_CAT": 2}]
        ).set_index("userid")

        converted = normalize_sce_raw_frame(raw, demographic_lookup=lookup)

        self.assertTrue(
            pd.isna(converted.loc[0, "sce_personal_job_loss_probability_1y"])
        )
        self.assertNotEqual(converted.loc[0, "income_group"], "unknown")


if __name__ == "__main__":
    unittest.main()
