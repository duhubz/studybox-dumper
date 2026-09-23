"""Machine-readable sidecars and human-readable run reports.

The decoder emits a JSON report per side and a short text summary. Reports are
the durable evidence for a run: they record the capture identity, clock
estimate, per-page page ids/offsets, per-packet-kind checksum counts, and the
dropout map, so a result can be reviewed without re-running the decoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__, framing


def page_dict(page: framing.DecodedPage) -> dict[str, Any]:
    by_kind: dict[str, dict[str, int]] = {}
    for packet in page.packets:
        if not packet.checked:
            continue
        entry = by_kind.setdefault(packet.kind, {"ok": 0, "total": 0})
        entry["total"] += 1
        entry["ok"] += 1 if packet.checksum_ok else 0
    return {
        "page_id": page.page_id,
        "lead_in_sample": page.lead_in_sample,
        "audio_sample": page.audio_sample,
        "byte_length": page.byte_length,
        "desync": page.desync,
        "complete": page.complete,
        "observed_coverage": page.observed_coverage,
        "packets": by_kind,
        "checksum_ok": page.checksum_ok,
        "checksum_total": page.checksum_total,
        "data128_ok": len([p for p in page.data128 if p.checksum_ok]),
        "data128_total": len(page.data128),
    }


def decode_report(result: framing.DecodeResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tool": "studybox.decode",
        "version": __version__,
        "capture": result.capture,
        "sample_rate": result.sample_rate,
        "half_cell_us": result.half_cell * 1e6,
        "transitions": result.transitions,
        "malformed": result.malformed,
        "pages": [page_dict(page) for page in result.pages],
        "totals": {
            "pages": len(result.pages),
            "checksum_ok": result.checksum_ok,
            "checksum_total": result.checksum_total,
            "checksum_rate": result.checksum_rate,
            "data128_ok": result.data128_ok,
            "data128_total": result.data128_total,
        },
        "dropouts": result.unrecoverable,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_text(report: dict[str, Any]) -> str:
    totals = report.get("totals", {})
    capture = report.get("capture", {})
    name = capture.get("name", "capture")
    lines = [
        f"{name}: {totals.get('pages', 0)} page(s), "
        f"checksum {totals.get('checksum_ok', 0)}/{totals.get('checksum_total', 0)} "
        f"({totals.get('checksum_rate', 0.0):.1%}), "
        f"128B {totals.get('data128_ok', 0)}/{totals.get('data128_total', 0)}, "
        f"half-cell {report.get('half_cell_us', 0.0):.2f} us",
    ]
    for page in report.get("pages", []):
        lines.append(
            f"  page {page['page_id']}: lead_in={page['lead_in_sample']} "
            f"audio={page['audio_sample']} bytes={page['byte_length']} "
            f"cksum={page['checksum_ok']}/{page['checksum_total']} "
            f"128B={page['data128_ok']}/{page['data128_total']}")
    dropouts = report.get("dropouts", [])
    if dropouts:
        lines.append(f"  dropouts: {len(dropouts)}")
        for item in dropouts[:8]:
            lines.append(f"    {item}")
    return "\n".join(lines)
