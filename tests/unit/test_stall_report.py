import numpy as np

from megabake.runtime import STALL_WORDS, format_stall_report, stall_records


def _region(workers, rows):
    region = np.zeros(workers * STALL_WORDS, dtype=np.int64)
    for cta, (call, kind, phase, counter, target, observed) in rows.items():
        region[cta * STALL_WORDS : (cta + 1) * STALL_WORDS] = (
            1, call, kind, phase, counter, target, observed, 123
        )
    return region


def test_stall_records_skip_ctas_without_a_report():
    records = stall_records(_region(4, {1: (0, 1, 14, 7, 480, 479)}))
    assert records == [{
        "cta": 1, "call": 0, "kind": "grid barrier", "phase": 14, "counter": 7,
        "target": 480, "observed": 479, "time_ns": 123,
    }]


def test_report_groups_ctas_by_wait_and_names_silent_ctas_and_phases():
    rows = {cta: (0, 1, 14, 7, 480, 479) for cta in (0, 1, 2, 5)}
    rows[4] = (0, 2, -1, 9, 3, 2)
    text = format_stall_report(stall_records(_region(7, rows)), 7, {14: "g0_range (region:mm_3)"})
    lines = text.splitlines()
    assert lines[0] == "MegaBake stall report: 5 of 7 CTAs timed out waiting"
    barrier = next(line for line in lines if "grid barrier" in line)
    assert barrier.split()[-5:] == ["0-2,5", "(closes", "after", "g0_range", "(region:mm_3))"]
    exchange = next(line for line in lines if "attention split exchange" in line)
    assert exchange.split()[-1] == "4"
    assert lines[-1].startswith("  no report from CTAs 3,6")


def test_report_without_records():
    assert format_stall_report([]) == "MegaBake stall report: no CTA reported a stall."
