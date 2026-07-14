import unittest

from fighter_rl.envs.loiter import CompetitionLoiterCurriculumEnv
from fighter_rl.training.sac import stage_selected
from fighter_rl.training.stages import advancement_satisfied, load_stages


class MicroCurriculumTest(unittest.TestCase):
    def test_legacy_schedule_is_unchanged_and_micro_schedule_is_contiguous(self):
        legacy = load_stages(schedule="gun_bucket_curriculum")
        micro = load_stages(schedule="gun_micro_curriculum")

        self.assertEqual(len(legacy), 17)
        self.assertEqual(len(micro), 29)
        self.assertEqual([stage.index for stage in micro], list(range(29)))
        self.assertEqual([stage.name for stage in micro[:6]], [stage.name for stage in legacy[:6]])
        self.assertEqual(len({stage.name for stage in micro}), len(micro))

    def test_reward_is_fixed_within_blocks_and_changes_at_boundaries(self):
        stages = load_stages(schedule="gun_micro_curriculum")

        self.assertEqual(stages[6].reward, stages[12].reward)
        self.assertEqual(stages[13].reward, stages[19].reward)
        self.assertNotEqual(stages[12].reward, stages[13].reward)
        self.assertGreater(stages[6].reward["wez_hold_scale"], 0.0)
        self.assertGreater(stages[13].reward["wez_hold_scale"], 0.0)

    def test_bucket_tail_worst_metrics_have_correct_direction(self):
        env = object.__new__(CompetitionLoiterCurriculumEnv)
        env.completed = [
            {
                "episodes": 2.0,
                "bucket_metrics": {
                    "easy": {
                        "episodes": 1.0,
                        "tail_track_score": 0.8,
                        "tail_wez_fraction": 0.2,
                        "tail_ata_deg": 4.0,
                    },
                    "hard": {
                        "episodes": 1.0,
                        "tail_track_score": 0.4,
                        "tail_wez_fraction": 0.05,
                        "tail_ata_deg": 12.0,
                    },
                },
            }
        ]

        summary = env.pop_completed_summary()

        self.assertEqual(summary["bucket_worst_tail_track_score"], 0.4)
        self.assertEqual(summary["bucket_worst_tail_wez_fraction"], 0.05)
        self.assertEqual(summary["bucket_worst_tail_ata_deg"], 12.0)

    def test_name_based_reset_selection_and_gate(self):
        stage = load_stages(schedule="gun_micro_curriculum")[13]

        self.assertTrue(stage_selected(stage, 13, names=["G0_static_wez_hold"]))
        self.assertFalse(stage_selected(stage, 13, names=["A0_nose_stabilize"]))
        self.assertTrue(stage_selected(stage, 13, indices=[13]))

        metrics = {
            key.removesuffix("_min").removesuffix("_max"): value
            for key, value in stage.advance_conditions.items()
        }
        ok, _ = advancement_satisfied(stage, metrics)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
