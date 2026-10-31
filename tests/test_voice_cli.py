import json
from pathlib import Path

from typer.testing import CliRunner

from colophon.cli import app
from colophon.voice.export import build_voice_prompt
from colophon.voice.models import VoiceProfile

CORPUS = Path(__file__).parents[1] / "examples" / "voice-corpus" / "corpus.json"
runner = CliRunner()


def test_real_cli_extract_readback_export_and_prompt(tmp_path):
    output = tmp_path / "voice.json"
    result = runner.invoke(app, ["voice", "extract", str(CORPUS), "--author", "Rowan (synthetic)", "-o", str(output)])
    assert result.exit_code == 0, result.output
    profile = VoiceProfile.model_validate_json(output.read_text())
    assert {document.entry.id for document in profile.documents} == {"rowan-harbor", "rowan-library"}
    assert "rowan-garden" not in {span.document_id for span in profile.exemplars}
    assert all(document.measurement.schema_version == "2" for document in profile.documents)
    assert profile.traits and all(trait.evidence for trait in profile.traits)
    inspection = runner.invoke(app, ["voice", "inspect", str(output)])
    assert inspection.exit_code == 0
    assert json.loads(inspection.output)["profile_id"] == profile.profile_id
    exported = tmp_path / "voice.md"
    result = runner.invoke(app, ["voice", "export", str(output), "-o", str(exported)])
    assert result.exit_code == 0, result.output
    assert "characters" in exported.read_text()
    validation = tmp_path / "stability.json"
    checked = runner.invoke(app, ["voice", "evaluate-profile", str(output), str(CORPUS), "-o", str(validation)])
    assert checked.exit_code == 0, checked.output
    stability = json.loads(validation.read_text())
    assert stability["evaluation"] == "single_author_held_out_stability"
    assert stability["measurement_coverage"] == {"total": 1, "measured": 1, "errors": 0}
    for mode in ["numeric", "exemplars", "qualitative", "hybrid"]:
        result = runner.invoke(app, ["voice", "prompt", str(output), "--mode", mode])
        assert result.exit_code == 0, result.output
        assert "Do not invent" in result.output
        assert "working-class" not in result.output
        generated = tmp_path / f"{mode}.generate.txt"
        result = runner.invoke(app, [
            "voice", "prompt", str(output), "--mode", mode, "--task", "generate", "-o", str(generated)
        ])
        assert result.exit_code == 0, result.output
        assert generated.read_text() == build_voice_prompt(profile, mode, task="generate") + "\n"
    invalid = runner.invoke(app, ["voice", "prompt", str(output), "--task", "unknown"])
    assert invalid.exit_code == 1
    assert "Unknown voice prompt task" in invalid.output


def test_cli_json_stdout_and_failed_extract_preserves_existing_artifact(tmp_path):
    result = runner.invoke(app, ["corpus", "validate", str(CORPUS)])
    assert result.exit_code == 0
    assert json.loads(result.output)["included_documents"] == 7
    output = tmp_path / "existing.json"
    output.write_text("retained artifact")
    result = runner.invoke(app, ["voice", "extract", str(CORPUS), "--author", "Missing", "-o", str(output)])
    assert result.exit_code == 1
    assert "No included training" in result.output
    assert output.read_text() == "retained artifact"


def test_cli_evaluation_readback_has_independent_channels(tmp_path):
    cases = CORPUS.with_name("writing-cases.json")
    output = tmp_path / "writing.json"
    result = runner.invoke(app, ["voice", "evaluate-writing", str(cases), "-o", str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    records = {record["id"]: record for record in report["records"]}
    assert records["faithful"]["channels"]["preservation_flags"]["value"] == 1
    assert records["changed-number"]["channels"]["preservation_flags"]["value"] == 0
    assert records["copied"]["channels"]["copying"]["value"] == 1
    assert records["faithful"]["channels"]["naturalness"]["status"] == "unscored"
