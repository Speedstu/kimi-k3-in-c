import json
import tempfile
import unittest
from pathlib import Path

from compact.data_contract import normalized_text_hash
from compact.full_train_preflight import run_preflight


ROOT = Path(__file__).resolve().parents[2]


def write_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    for i, domain in enumerate(("code", "agentic", "cyber", "general")):
        prompt = f"clean training prompt {domain} {i}"
        rows.append(
            {
                "sample_id": f"sample-{domain}",
                "domain": domain,
                "split": "train",
                "source_id": f"synthetic/{domain}/{i}",
                "prompt_sha256": normalized_text_hash(prompt),
                "chosen_sha256": normalized_text_hash(prompt + " chosen"),
                "rejected_sha256": normalized_text_hash(prompt + " rejected"),
                "verifier_id": f"synthetic-{domain}:v1",
                "verifier_passed": True,
            }
        )
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    return rows


class K3CompactFullTrainPreflightTests(unittest.TestCase):
    def fixture(self, td: Path):
        model = td / "teacher"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        # Tiny fixture: tests override the full-checkpoint byte/shard thresholds.
        (model / "model-00001.safetensors").write_bytes(b"teacher-fixture")
        manifest = td / "train.jsonl"
        rows = write_manifest(manifest)
        denied = td / "heldout.sha256"
        denied.write_text(normalized_text_hash("totally separate heldout item") + "\n", encoding="utf-8")
        output = td / "out"
        return model, manifest, denied, output, rows

    def run_fixture(self, td: Path, **kwargs):
        model, manifest, denied, output, rows = self.fixture(td)
        report = run_preflight(
            model_dir=model,
            train_manifest=manifest,
            heldout_hashes=denied,
            output_dir=output,
            spec_path=ROOT / "compact/k3_compact.json",
            min_teacher_bytes=1,
            min_shards=1,
            min_free_output_bytes=1,
            **kwargs,
        )
        return report, (model, manifest, denied, output, rows)

    def test_clean_fixture_passes_without_claiming_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self.run_fixture(Path(tmp))
            self.assertTrue(report["pass"])
            self.assertEqual(report["data"]["total"], 4)
            self.assertGreater(report["architecture"]["estimated_checkpoint_gb"], 80.0)
            self.assertLess(report["architecture"]["estimated_checkpoint_gb"], 100.0)
            self.assertIn("preflight only", report["claim"])

    def test_missing_full_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            model, manifest, denied, output, _ = self.fixture(td)
            (model / "model-00001.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "no .safetensors"):
                run_preflight(
                    model_dir=model,
                    train_manifest=manifest,
                    heldout_hashes=denied,
                    output_dir=output,
                    spec_path=ROOT / "compact/k3_compact.json",
                    min_teacher_bytes=1,
                    min_shards=1,
                    min_free_output_bytes=1,
                )

    def test_heldout_collision_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            model, manifest, denied, output, rows = self.fixture(td)
            denied.write_text(str(rows[0]["prompt_sha256"]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collides with held-out"):
                run_preflight(
                    model_dir=model,
                    train_manifest=manifest,
                    heldout_hashes=denied,
                    output_dir=output,
                    spec_path=ROOT / "compact/k3_compact.json",
                    min_teacher_bytes=1,
                    min_shards=1,
                    min_free_output_bytes=1,
                )

    def test_missing_specialist_domain_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            model, manifest, denied, output, rows = self.fixture(td)
            kept = [x for x in rows if x["domain"] != "cyber"]
            manifest.write_text("".join(json.dumps(x) + "\n" for x in kept), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero 'cyber' records"):
                run_preflight(
                    model_dir=model,
                    train_manifest=manifest,
                    heldout_hashes=denied,
                    output_dir=output,
                    spec_path=ROOT / "compact/k3_compact.json",
                    min_teacher_bytes=1,
                    min_shards=1,
                    min_free_output_bytes=1,
                )

    def test_cuda_requirement_fails_without_enough_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            model, manifest, denied, output, _ = self.fixture(td)
            with self.assertRaisesRegex(ValueError, "CUDA devices|PyTorch is required"):
                run_preflight(
                    model_dir=model,
                    train_manifest=manifest,
                    heldout_hashes=denied,
                    output_dir=output,
                    spec_path=ROOT / "compact/k3_compact.json",
                    min_teacher_bytes=1,
                    min_shards=1,
                    min_free_output_bytes=1,
                    min_cuda_devices=10_000,
                )


if __name__ == "__main__":
    unittest.main()
