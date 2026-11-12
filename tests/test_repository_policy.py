"""Policy checks use generated in-memory data only."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "repository_policy", Path(__file__).resolve().parents[1] / "scripts/check_repository.py")
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


class RepositoryPolicyTests(unittest.TestCase):
    def test_code_and_docs(self):
        for name in ("README.md", "src/model.py", "configs/demo.yaml", ".gitignore"):
            self.assertEqual(policy.inspect_content(name, b"# example\n"), [])

    def test_artifact_extension(self):
        self.assertTrue(policy.inspect_content("demo.mp4", b"example"))

    def test_runtime_folder(self):
        self.assertTrue(policy.inspect_content("outputs/episode.json", b"{}"))

    def test_environment_file(self):
        self.assertTrue(policy.inspect_content(".env", b"EXAMPLE=value"))

    def test_size(self):
        self.assertTrue(policy.inspect_content("large.md", b"x" * (policy.LIMIT + 1)))

    def test_binary(self):
        self.assertTrue(policy.inspect_content("binary.md", bytes([0, 1, 2])))

    def test_generated_secret(self):
        sample = ("gh" + "p_" + "a" * 36).encode()
        self.assertTrue(policy.inspect_content("config.txt", sample))

    def test_utf8(self):
        self.assertEqual(policy.inspect_content("说明.md", "实验规范".encode()), [])


if __name__ == "__main__":
    unittest.main()
