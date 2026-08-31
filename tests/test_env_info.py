"""Unit tests for run-environment capture.

Kaggle resolves its own package versions, so every run must record what it
actually ran with; this is what makes a result reproducible after the fact.
"""

from src.utils.env_info import collect_env


class TestCollectEnv:
    def test_records_the_libraries_that_decide_numerical_results(self):
        env = collect_env()
        assert {"python", "torch", "numpy", "transformers", "peft",
                "datasets"} <= set(env)

    def test_versions_are_concrete_strings(self):
        env = collect_env()
        assert all(isinstance(v, str) and v for v in env.values())

    def test_records_gpu_and_cuda_availability(self):
        env = collect_env()
        assert "cuda_available" in env
        assert "device_name" in env

    def test_records_the_git_commit_for_run_identification(self):
        # CLAUDE.md section 8: run id = timestamp + git commit hash.
        assert "git_commit" in collect_env()

    def test_missing_library_is_reported_not_raised(self, monkeypatch):
        import src.utils.env_info as env_info

        def boom(name):
            raise ImportError(name)

        monkeypatch.setattr(env_info, "_version_of", boom)
        assert collect_env()["torch"] == "not installed"
