from megabake.machine import DeviceFingerprint, MachineModel


def _model() -> MachineModel:
    return MachineModel(
        fingerprint=DeviceFingerprint("gpu", 60, "9.0", 1, 1, 1, 1, 1),
        hbm_read_bytes_per_s=2000e9,
        hbm_read_bytes_per_s_by_ctas={1: 80e9, 30: 1800e9, 60: 2000e9},
        l2_read_bytes_per_s=4000e9,
        l2_effective_bytes=32 << 20,
        read_bytes_per_s_by_footprint={1 << 20: 3500e9},
        grid_barrier_s_by_ctas={1: 0.5e-6, 60: 0.7e-6},
        counter_round_trip_s=0.9e-6,
        tensor_peak_flops={"bfloat16": 400e12},
    )


def test_read_bandwidth_interpolates_and_saturates():
    model = _model()
    assert model.read_bandwidth(1) == 80e9
    assert model.read_bandwidth(30) == 1800e9
    assert 80e9 < model.read_bandwidth(15) < 1800e9
    assert model.read_bandwidth(60) == model.read_bandwidth(None) == 2000e9
    assert model.read_bandwidth(200) == 2000e9


def test_speed_of_light_takes_the_binding_limit():
    model = _model()
    assert model.speed_of_light_s(2000e9) == 1.0
    assert model.speed_of_light_s(0, 400e12) == 1.0
    assert model.grid_barrier_s(50) == 0.7e-6


def test_json_round_trip():
    model = _model()
    assert MachineModel.from_json(model.to_json()) == model
