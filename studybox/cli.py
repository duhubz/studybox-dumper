"""Command line entry points: ``python3 -m studybox decode|verify|report|roundtrip|merge``.

``decode`` turns a capture into page/checksum diagnostics and (optionally) a
``.studybox`` container plus a JSON sidecar. ``verify`` runs the independent
decode gate over a capture or an existing container. ``report`` renders a saved
decode sidecar. ``roundtrip`` runs the bit-level MFM round-trip gate. ``merge``
fills dropout pages from a second recording of the same program.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, api, audio_in, encoder, merge, paths, report, verify


def cmd_decode(args: argparse.Namespace) -> int:
    outcome = api.decode_file(args.target, args.side, args.channel,
                              seconds=args.seconds)
    result = outcome.result
    if args.studybox and not result.pages:
        print("decode: no pages decoded; refusing to write a .studybox",
              file=sys.stderr)
        return 2
    payload = report.decode_report(result)
    # Preflight every destination together before writing any of them: a JSON
    # sidecar must not alias the capture, the container, or the container path.
    api.require_distinct_outputs(
        [("JSON sidecar", args.json), ("container output", args.studybox)],
        outcome.capture)
    if args.json:
        report.write_json(args.json, payload)
    if args.studybox:
        written, page_count, audio_bytes = api.write_studybox(
            args.studybox, result, outcome.capture, no_audio=args.no_audio)
        print(f"wrote {written} ({page_count} page(s), {audio_bytes} audio bytes)")
    if not args.quiet:
        print(report.render_text(payload))
    if args.print_json:
        import json
        print(json.dumps(payload, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    if args.studybox:
        paths.require_distinct([("JSON sidecar", args.json)],
                               [("studybox input", args.studybox)])
        try:
            gate = verify.verify_studybox_file(args.studybox, args.min_checksum_rate,
                                               args.allow_degraded)
        except (OSError, ValueError) as error:
            print(f"verify: {error}", file=sys.stderr)
            return 2
    elif args.target:
        capture = api.resolve_target(args.target, args.side, args.channel)
        paths.require_distinct([("JSON sidecar", args.json)],
                               api.capture_inputs(capture))
        gate = api.verify_file(capture, None, args.channel, args.seconds,
                               args.min_checksum_rate, args.allow_degraded)
    else:
        print("verify: provide a target or --studybox", file=sys.stderr)
        return 2
    if args.json:
        report.write_json(args.json, gate.to_dict())
    print(gate.render())
    return 0 if gate.passed else 1


def cmd_report(args: argparse.Namespace) -> int:
    paths.require_distinct([("text report", args.out)],
                           [("JSON sidecar", args.json)])
    payload = report.load_json(args.json)
    text = report.render_text(payload)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    capture = api.resolve_target(args.target, args.side, args.channel)
    paths.require_distinct([("JSON sidecar", args.json)],
                           api.capture_inputs(capture))
    outcome = api.decode_file(capture, args.side, args.channel,
                              seconds=args.seconds)
    result = outcome.result
    trips = [(page, encoder.roundtrip_page(page)) for page in result.pages]
    exact = [trip for _, trip in trips if trip.exact]
    payload = [trip for _, trip in trips if trip.payload_exact]
    data = [trip for _, trip in trips if trip.payload_data_exact]
    tail = [trip for _, trip in trips if trip.tail_dropout]
    glitch = [trip for _, trip in trips if not trip.payload_exact
              and trip.payload_data_exact]
    incomplete = [page for page, _ in trips if not page.complete]
    bad_checksums = [page for page, _ in trips
                     if page.checksum_ok != page.checksum_total]
    covered = len(trips) > 0
    exact_all = covered and len(exact) == len(trips)
    # Strict success requires retained-page exactness AND independent
    # verification of the raw bytes. Exact retained pages alone do not prove
    # capture coverage: a damaged page can be discarded as a tail while every
    # page that did decode re-encodes exactly. The verifier re-parses the raw
    # bytes, so it also catches capture losses, desync, and malformed packets.
    gate = verify.verify_decode_result(result)
    losses = [item for item in result.unrecoverable
              if item.get("kind") not in verify.BENIGN_DROPOUT_KINDS]
    strict_ok = (exact_all and not incomplete and not bad_checksums
                 and gate.passed)
    payload_json = {
        "schema_version": 1,
        "tool": "studybox.roundtrip",
        "version": __version__,
        "capture": result.capture,
        "pages": len(trips),
        "strict": strict_ok,
        "exact_pages": len(exact),
        "exact": exact_all,
        "complete": covered and not incomplete,
        "incomplete_pages": len(incomplete),
        "checksums_ok": covered and not bad_checksums,
        "bad_checksum_pages": len(bad_checksums),
        "payload_exact_pages": len(payload),
        "payload_exact": covered and len(payload) == len(trips),
        "payload_data_exact_pages": len(data),
        "payload_data_exact": covered and len(data) == len(trips),
        "tail_dropout_pages": len(tail),
        "clock_glitch_pages": len(glitch),
        "verify_passed": gate.passed,
        "verification": gate.to_dict(),
        "capture_losses": result.unrecoverable,
        "results": [trip.to_dict() for _, trip in trips],
    }
    if args.json:
        report.write_json(args.json, payload_json)
    print(f"roundtrip: {'PASS' if strict_ok else 'FAIL'} "
          f"({len(exact)}/{len(trips)} exact, {len(incomplete)} incomplete, "
          f"{len(bad_checksums)} checksum-failed; verify "
          f"{'PASS' if gate.passed else 'FAIL'}; diagnostic "
          f"{len(data)}/{len(trips)} payload data bit-exact, "
          f"{len(tail)} tail dropout(s), {len(glitch)} clock glitch page(s))")
    for page, trip in trips:
        if not page.complete:
            print(f"  page {trip.page_id}: INCOMPLETE (no valid end control)")
        elif page.checksum_ok != page.checksum_total:
            print(f"  page {trip.page_id}: CHECKSUM {page.checksum_ok}/"
                  f"{page.checksum_total} did not pass")
        elif not trip.exact:
            print(f"  page {trip.page_id}: NOT EXACT missing={trip.missing} "
                  f"extra={trip.extra}")
        elif trip.tail_dropout:
            print(f"  page {trip.page_id}: tail dropout (extra={trip.extra} "
                  f"missing={trip.missing}, {trip.tail_bytes} tail byte(s))")
        elif not trip.payload_exact:
            print(f"  page {trip.page_id}: clock glitch ({trip.clock_glitches} "
                  f"clock edge(s), payload data intact)")
    if losses:
        print(f"  capture losses: {len(losses)}")
        for item in losses[:8]:
            print(f"    {item}")
    if not gate.passed:
        for check in gate.checks:
            if not check.passed:
                print(f"  verify [{check.name}]: {check.detail}")
    return 0 if strict_ok else 1


def cmd_merge(args: argparse.Namespace) -> int:
    inputs = [("base container", args.base)]
    inputs += [(f"other container {index}", path)
               for index, path in enumerate(args.others, start=1)]
    paths.require_distinct(
        [("JSON sidecar", args.json), ("container output", args.studybox)],
        inputs)
    try:
        outcome = merge.merge_files(args.base, args.others)
    except (OSError, ValueError) as error:
        print(f"merge: {error}", file=sys.stderr)
        return 2
    if args.json:
        report.write_json(args.json, outcome.provenance)
    outcome.box.write(args.studybox)
    print(f"wrote {args.studybox} ({len(outcome.box.pages)} page(s), "
          f"{outcome.provenance['repaired']} repaired, "
          f"{outcome.provenance['open_conflicts']} open conflict(s))")
    return 0


def console_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studybox", description=__doc__)
    parser.add_argument("--version", action="version", version=f"studybox {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_target(sub_parser: argparse.ArgumentParser, required: bool) -> None:
        sub_parser.add_argument("target", nargs=None if required else "?",
                                help="capture directory (with --side) or a file")
        sub_parser.add_argument("--side", choices=["A", "B"],
                                help="side of a directory dump following the "
                                     "casa/casb naming convention")
        sub_parser.add_argument("--data-channel", "--channel", dest="channel",
                                type=int, default=audio_in.DATA_CHANNEL,
                                metavar="N",
                                help="data channel for an explicit stereo file "
                                     "(default: %(default)s, the right channel)")
        sub_parser.add_argument("--seconds", type=float, default=None,
                                help="decode only the first N seconds for a quick "
                                     "diagnostic; clock and page boundaries are "
                                     "estimated from that partial signal, so "
                                     "results can differ from a full decode")

    decode = sub.add_parser("decode", help="decode a capture and report pages/checksums")
    add_target(decode, True)
    decode.add_argument("--json", help="write the JSON sidecar here")
    decode.add_argument("--studybox", help="also write a .studybox container here")
    decode.add_argument("--no-audio", action="store_true",
                        help="embed silence when no narration track is present "
                             "instead of failing")
    decode.add_argument("--print-json", action="store_true", help="print the JSON sidecar")
    decode.add_argument("--quiet", action="store_true")
    decode.set_defaults(func=cmd_decode)

    check = sub.add_parser("verify", help="run the independent decode gate")
    add_target(check, False)
    check.add_argument("--studybox", help="verify an existing .studybox file")
    check.add_argument("--min-checksum-rate", type=float,
                       default=verify.DEFAULT_MIN_CHECKSUM_RATE)
    check.add_argument("--allow-degraded", action="store_true",
                       help="accept dropouts/failed checksums under the rate "
                            "threshold; the result is labelled degraded")
    check.add_argument("--json", help="write the verification report here")
    check.set_defaults(func=cmd_verify)

    render = sub.add_parser("report", help="render a saved decode sidecar")
    render.add_argument("json", help="decode JSON sidecar")
    render.add_argument("--out", help="also write the text report here")
    render.set_defaults(func=cmd_report)

    roundtrip = sub.add_parser("roundtrip", help="bit-level MFM round-trip gate")
    add_target(roundtrip, True)
    roundtrip.add_argument("--json", help="write the round-trip report here")
    roundtrip.set_defaults(func=cmd_roundtrip)

    combine = sub.add_parser(
        "merge", help="fill dropout pages from another recording of the same program")
    combine.add_argument("base", help="container providing the audio and page offsets")
    combine.add_argument("others", nargs="+",
                         help="container(s) from another recording of the same program")
    combine.add_argument("--studybox", required=True,
                         help="write the merged container here")
    combine.add_argument("--json", help="write the per-page merge provenance here")
    combine.set_defaults(func=cmd_merge)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = console_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, audio_in.sf.LibsndfileError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
