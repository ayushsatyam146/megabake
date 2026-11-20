from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from megabake import doctor
from megabake.main import main


def _statuses(checks):
    return {check.name: check.status for check in checks}


def _fake_cuda_machine(monkeypatch, *, capability, nvcc_release):
    monkeypatch.setattr(doctor.torch.version, "cuda", "13.0")
    monkeypatch.setattr(doctor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(doctor.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(doctor.torch.cuda, "get_device_capability", lambda index: capability)
    monkeypatch.setattr(doctor.torch.cuda, "get_device_name", lambda index: "Fake GPU")
    monkeypatch.setattr(doctor, "_find_nvcc", lambda: "/fake/nvcc")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"Cuda compilation tools, release {nvcc_release}, V{nvcc_release}.1"
        ),
    )


def _cpu_only_machine(monkeypatch):
    monkeypatch.setattr(doctor.torch.version, "cuda", None)
    monkeypatch.setattr(doctor.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(doctor, "_find_nvcc", lambda: None)


def test_cpu_only_machine_warns_by_default_and_fails_when_cuda_required(monkeypatch):
    _cpu_only_machine(monkeypatch)

    relaxed = _statuses(doctor.run_checks())
    strict = _statuses(doctor.run_checks(require_cuda=True))

    for name in ("torch CUDA build", "GPU", "nvcc"):
        assert relaxed[name] == "warn"
        assert strict[name] == "fail"


def test_supported_gpu_and_toolkit_pass(monkeypatch):
    _fake_cuda_machine(monkeypatch, capability=(9, 0), nvcc_release="13.0")

    statuses = _statuses(doctor.run_checks(require_cuda=True))

    assert statuses["torch CUDA build"] == "ok"
    assert statuses["GPU"] == "ok"
    assert statuses["nvcc"] == "ok"


@pytest.mark.parametrize(
    ("capability", "nvcc_release", "failing"),
    [((8, 0), "13.0", "GPU"), ((9, 0), "12.8", "nvcc")],
)
def test_unsupported_gpu_or_toolkit_is_reported(monkeypatch, capability, nvcc_release, failing):
    _fake_cuda_machine(monkeypatch, capability=capability, nvcc_release=nvcc_release)

    checks = {check.name: check for check in doctor.run_checks(require_cuda=True)}

    assert checks[failing].status == "fail"
    assert "need" in checks[failing].detail


def test_nvcc_that_fails_to_run_is_reported(monkeypatch):
    monkeypatch.setattr(doctor, "_find_nvcc", lambda: "/fake/nvcc")

    def broken_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvcc")

    monkeypatch.setattr(doctor.subprocess, "run", broken_run)

    assert doctor._check_nvcc("warn").status == "warn"


def test_find_nvcc_prefers_cuda_home(tmp_path, monkeypatch):
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))

    assert doctor._find_nvcc() == str(nvcc)


def test_doctor_command_exit_code_follows_failures(monkeypatch, capsys):
    _cpu_only_machine(monkeypatch)

    assert main(["doctor"]) == 0
    assert main(["doctor", "--cuda"]) == 1
    assert "[fail] GPU" in capsys.readouterr().out


def test_version_comes_from_package_metadata(monkeypatch, capsys):
    monkeypatch.setattr("megabake.main.version", lambda name: "9.9.9")

    with pytest.raises(SystemExit):
        main(["--version"])

    assert capsys.readouterr().out.strip() == "megabake 9.9.9"
