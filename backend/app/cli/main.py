"""CallSense command line interface.

Runs the same engine the API uses, so a result produced here is byte-identical
to one produced through HTTP. That is the property the old pipeline lacked:
its CLI output could not be read back by the API at all (issue #23).

    python -m app.cli.main analyse call.txt
    python -m app.cli.main import ./transcripts --owner alice
    python -m app.cli.main keygen
    python -m app.cli.main evaluate labelled.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.analysis.engine import AnalysisEngine
from app.core.errors import CallSenseError
from app.domain.models import CallAnalysis

SUPPORTED_SUFFIXES = {".txt", ".json", ".vtt", ".srt", ".md"}

_BAND_COLOUR = {
    "hot": "\033[91m", "warm": "\033[93m", "cold": "\033[94m",
    "disqualified": "\033[90m",
}
_RESET = "\033[0m"


def _colour(text: str, band: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_BAND_COLOUR.get(band, '')}{text}{_RESET}"


def render_analysis(analysis: CallAnalysis, colour: bool = True) -> str:
    """Human-readable report for a terminal."""
    score = analysis.lead_score
    lines = [
        "",
        _colour(f"  {analysis.summary.headline}", score.band.value, colour),
        f"  {'─' * 66}",
        f"  Score       {score.score:.1f}/100  ({score.band.value}, "
        f"confidence {score.confidence:.0%})",
        f"  Outcome     {analysis.outcome.label.value} "
        f"({analysis.outcome.confidence:.0%})",
        f"  Sentiment   {analysis.sentiment.overall:+.2f} overall, "
        f"trend {analysis.sentiment.trend:+.2f}",
    ]

    talk = analysis.talk_metrics
    if talk.prospect_talk_ratio:
        lines.append(
            f"  Talk ratio  rep {talk.rep_talk_ratio:.0%} / "
            f"prospect {talk.prospect_talk_ratio:.0%}"
        )

    lines += ["", "  SCORE BREAKDOWN"]
    for component in score.components:
        bar = "█" * int(component.raw * 20)
        lines.append(
            f"    {component.name:<12} {component.contribution:5.1f}/"
            f"{component.weight:<5.0f} {bar:<20} {component.detail}"
        )
    for penalty in score.penalties:
        lines.append(f"    {penalty.name:<12} {penalty.contribution:5.1f}        "
                     f"{'':<20} {penalty.detail}")

    if analysis.signals:
        lines += ["", "  SIGNALS"]
        lines += [
            f"    {s.type.value:<15} {s.strength:.2f}  "
            f'"{s.evidence[0].quote[:60]}"' if s.evidence
            else f"    {s.type.value:<15} {s.strength:.2f}"
            for s in analysis.signals[:6]
        ]

    if analysis.objections:
        lines += ["", "  OBJECTIONS"]
        for objection in analysis.objections:
            flag = "addressed" if objection.addressed else "UNADDRESSED"
            lines.append(
                f"    {objection.category.value:<12} {objection.severity:.2f}  [{flag}]"
            )

    lines += ["", "  NEXT ACTIONS"]
    for action in analysis.next_actions:
        lines.append(
            f"    [{action.priority.value.upper()}] {action.action} "
            f"(within {action.due_within_hours}h)"
        )

    if analysis.metadata.warnings:
        lines += ["", "  WARNINGS"]
        lines += [f"    ! {w}" for w in analysis.metadata.warnings]

    lines.append("")
    return "\n".join(lines)


def cmd_analyse(args: argparse.Namespace) -> int:
    """Analyse one transcript and print or dump the result."""
    path = Path(args.path)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    engine = AnalysisEngine()
    try:
        transcript, analysis = engine.analyse_text(
            path.read_bytes(), call_id=path.stem, filename=path.name
        )
    except CallSenseError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(analysis.model_dump(mode="json"), indent=2))
    else:
        print(f"  Source: {path.name} ({transcript.source_format}, "
              f"{transcript.word_count} words)")
        colour = sys.stdout.isatty() and not args.no_colour
        print(render_analysis(analysis, colour=colour))

    if args.out:
        Path(args.out).write_text(
            json.dumps(analysis.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        print(f"  Written to {args.out}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Bulk-ingest a directory of transcripts through the API's own service."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        print(f"No transcripts found in {directory} "
              f"(looked for {', '.join(sorted(SUPPORTED_SUFFIXES))})")
        return 0

    return asyncio.run(_import_all(files, args))


async def _import_all(files: list[Path], args: argparse.Namespace) -> int:
    from app.core.config import get_settings
    from app.db.repository import AuditRepository, CallRepository
    from app.db.session import create_schema, dispose_engine, init_engine, session_scope
    from app.services.ingestion import IngestionService, derive_title
    from app.storage.base import build_store

    settings = get_settings()
    init_engine(settings.database_url)
    await create_schema()

    engine = AnalysisEngine.from_settings(settings)
    blobs = build_store(settings)

    succeeded = failed = 0
    try:
        for path in files:
            try:
                async with session_scope() as session:
                    service = IngestionService(
                        CallRepository(session), blobs, engine,
                        AuditRepository(session),
                    )
                    call = await service.ingest(
                        content=path.read_bytes(),
                        title=derive_title(path.name),
                        principal_id="cli",
                        filename=path.name,
                        owner=args.owner,
                    )
                succeeded += 1
                print(f"  ✓ {path.name:<45} {call.lead_score:>5.1f} "
                      f"{call.lead_band.value if call.lead_band else '-'}")
            except Exception as exc:
                failed += 1
                print(f"  ✗ {path.name:<45} {exc}", file=sys.stderr)
    finally:
        await dispose_engine()

    print(f"\nImported {succeeded} call(s), {failed} failed.")
    return 1 if failed and not succeeded else 0


def cmd_keygen(args: argparse.Namespace) -> int:
    """Mint an API key."""
    from app.core.security import generate_api_key

    key = generate_api_key()
    print(f"\n  API key ({args.role}):\n\n    {key}\n")
    print("  Add it to your environment:\n")
    print(f'    AUTH_ENABLED=true\n    API_KEYS="{key}:{args.role}"\n')
    print("  Store it in a secret manager. It cannot be recovered from the "
          "server,\n  which only keeps a hash.\n")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Validate the scoring rubric against labelled outcomes."""
    from app.analysis.evaluation import evaluate

    path = Path(args.csv)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    print(evaluate(path).render())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_config=None,  # logging is configured by the app itself
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="callsense",
        description="Call intelligence for sales teams.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser(
        "analyse", aliases=["analyze"], help="Analyse a single transcript"
    )
    analyse.add_argument("path", help="Transcript file (.txt, .json, .vtt, .srt)")
    analyse.add_argument("--json", action="store_true", help="Emit raw JSON")
    analyse.add_argument("--out", help="Also write the JSON result to this path")
    analyse.add_argument("--no-colour", action="store_true")
    analyse.set_defaults(func=cmd_analyse)

    importer = subparsers.add_parser("import", help="Bulk-import a directory")
    importer.add_argument("directory")
    importer.add_argument("--owner", help="Assign all imported calls to this owner")
    importer.set_defaults(func=cmd_import)

    keygen = subparsers.add_parser("keygen", help="Generate an API key")
    keygen.add_argument(
        "--role", default="analyst", choices=["viewer", "analyst", "admin"]
    )
    keygen.set_defaults(func=cmd_keygen)

    evaluate_cmd = subparsers.add_parser(
        "evaluate", help="Score a labelled CSV and report calibration"
    )
    evaluate_cmd.add_argument("csv")
    evaluate_cmd.set_defaults(func=cmd_evaluate)

    serve = subparsers.add_parser("serve", help="Run the API server")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    return args.func(args)


def _force_utf8_output() -> None:
    """Make the reports printable when stdout is not a UTF-8 console.

    The reports use box-drawing and block characters. On Windows a *redirected*
    stream falls back to the legacy locale encoding (cp1252), which cannot
    encode them, so `callsense analyse call.txt > report.txt` -- and the
    quick-start command in README.md, under any piped shell -- died with
    UnicodeEncodeError before printing anything. ``errors="replace"`` keeps a
    terminal that genuinely cannot represent a glyph degrading to '?' instead
    of crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
