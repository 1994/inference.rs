"""Architecture gates prevent model logic from returning to orchestration."""

import unittest

from layout import check_ownership, production_dependencies


class OwnershipTests(unittest.TestCase):
    def test_allowed_dependencies(self):
        for member, deps in [
            ("crates/model/package", {"infer-model-recipes", "infer-spi"}),
            ("crates/model/compiler", {"infer-ir", "infer-kernel-api"}),
            ("crates/engine/runtime", {"infer-compiler", "infer-state"}),
        ]:
            check_ownership(member, deps)

    def test_model_and_device_logic_cannot_leak_into_core(self):
        for member, dependency in [
            ("crates/model/compiler", "infer-model-recipes"),
            ("crates/model/package", "infer-compiler"),
            ("crates/engine/runtime", "infer-models"),
            ("crates/engine/scheduler", "infer-kernel-api"),
            ("crates/engine/state", "infer-backend-cuda"),
            ("crates/foundation/spi", "infer-runtime"),
            ("crates/backend/kernel-api", "infer-backend-cuda"),
        ]:
            with self.subTest(member=member), self.assertRaises(SystemExit):
                check_ownership(member, {dependency})

    def test_target_specific_dependencies_cannot_bypass_gate(self):
        dependencies = production_dependencies(
            {
                "dev-dependencies": {"infer-models": {}},
                "target": {"cfg(unix)": {"dependencies": {"infer-model-recipes": {}}}},
            }
        )
        self.assertEqual(dependencies, {"infer-model-recipes"})
        with self.assertRaises(SystemExit):
            check_ownership("crates/engine/runtime", dependencies)


if __name__ == "__main__":
    unittest.main()
