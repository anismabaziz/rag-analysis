import argparse
import asyncio
from pathlib import Path

from core.registry import Architecture, UnknownArchitecture, all_architectures, architecture
from evaluation.run import RESULTS_DIR, run_evaluation
from evaluation.set import domains
from vector.store import reset_vector_store
from corpus.manifest import load_manifest
from corpus.populate import Outcome, populate, verify_corpus

# Where the corpus is populated. It is the same directory ingestion walks, so a document needs
# to be in no other place.
CORPUS_DIR = Path("documents")


async def main():
	"""
	Main CLI parser and router for the RAG Comparison Benchmarking application.
	Takes an architecture name for anything that reads or writes a collection, so which
	architectures exist is decided by what is registered and not by this file.
	"""

	# create main argument parser
	parser = argparse.ArgumentParser(
		prog="rag-analysis",
		description="RAG Comparison Benchmarking CLI - A tool to test and compare RAG architectures."
	)

	# add subparsers for each command
	subparsers = parser.add_subparsers(dest="command", required=True, help="RAG commands")

	# ingest command
	ingest_parser = subparsers.add_parser(
		"ingest",
		help="Reset an architecture's collection, load files from ./documents, split them, embed, and index into Qdrant"
	)
	ingest_parser.add_argument("architecture", type=str, help="The architecture to ingest the corpus for")

	# query command
	query_parser = subparsers.add_parser(
		"query",
		help="Retrieve context and answer a query using one architecture"
	)
	query_parser.add_argument("architecture", type=str, help="The architecture to answer with")
	query_parser.add_argument("question", type=str, help="The query/question to run")

	# run command
	run_parser = subparsers.add_parser(
		"run",
		help="Run one architecture over a domain's evaluation set and write the result file"
	)
	run_parser.add_argument("architecture", type=str, help="The architecture to measure")
	run_parser.add_argument(
		"--domain",
		required=True,
		help="The domain whose evaluation set to run, as named in evaluation/sets",
	)
	run_parser.add_argument(
		"--out",
		default=None,
		help="Where to write the result file. Defaults to results/runs/<domain>/",
	)

	# architectures command
	subparsers.add_parser(
		"architectures",
		help="List every registered architecture, with the collection it reads"
	)

	# summarize command
	summarize_parser = subparsers.add_parser(
		"summarize",
		help="Read every run file and print the pooled and per-domain retrieval table"
	)
	summarize_parser.add_argument(
		"--results-dir",
		default=None,
		help="Where the run files live. Defaults to results/runs/",
	)
	summarize_parser.add_argument(
		"--out",
		default=None,
		help="Write the markdown table to this file as well as printing it",
	)

	# clear command
	clear_parser = subparsers.add_parser(
		"clear",
		help="Clear/delete the Qdrant vector store collection to prevent data contamination"
	)
	clear_parser.add_argument("collection_name", type=str, help="The collection name to clear")

	# fetch-corpus command
	fetch_parser = subparsers.add_parser(
		"fetch-corpus",
		help="Download every document in the corpus manifest into ./documents, checking each digest"
	)
	fetch_parser.add_argument(
		"--domain",
		action="append",
		help="Fetch only this domain, as listed in the manifest. Repeatable. Both are fetched when left out."
	)

	# verify-corpus command
	verify_parser = subparsers.add_parser(
		"verify-corpus",
		help="Check the documents on disk against the manifest digests, without downloading anything"
	)
	verify_parser.add_argument(
		"--domain",
		action="append",
		help="Verify only this domain, as listed in the manifest. Repeatable. All are verified when left out."
	)

	# parse CLI arguments
	args = parser.parse_args()

	if args.command == "architectures":
		list_architectures()

	elif args.command == "ingest":
		print("[CLI] Ingesting documents...")

		resolve(args.architecture, parser).ingest()

	elif args.command == "query":
		print("[CLI] Running retrieval and generation...")

		await resolve(args.architecture, parser).build().answer(args.question)

	elif args.command == "run":
		print("[CLI] Running the evaluation set...")

		await run_command(args, parser)

	elif args.command == "summarize":
		print("[CLI] Summarizing the run files...")

		summarize_command(args, parser)

	elif args.command == "clear":
		print("[CLI] Clearing vector store...")

		reset_vector_store(args.collection_name)

	elif args.command in ("fetch-corpus", "verify-corpus"):
		corpus_command(args, parser)


def resolve(name: str, parser: argparse.ArgumentParser) -> Architecture:
	"""The named architecture, or a usage error naming what is registered instead.

	The name is resolved before anything is built or contacted, so a command naming an
	architecture that does not exist stops before it can touch a store.
	"""
	try:
		return architecture(name)
	except UnknownArchitecture as error:
		parser.error(str(error))


def list_architectures():
	"""Print every registered architecture, so a reader can see what exists without reading code."""
	print("[CLI] Registered architectures:")
	for registered in all_architectures():
		print(f"  {registered.name}\t{registered.collection}\t{registered.description}")


async def run_command(args, parser):
	"""Run one architecture over a domain's evaluation set, and say plainly when it cannot.

	The domain is checked against the sets that exist before the run starts, because a typo in
	`--domain` would otherwise be reported as a run in which retrieval found nothing.
	"""
	known = domains()
	if args.domain not in known:
		parser.error(f"no evaluation set for {args.domain}. There is one for: {', '.join(known)}")

	results_dir = Path(args.out) if args.out else RESULTS_DIR

	try:
		await run_evaluation(args.architecture, args.domain, results_dir=results_dir)
	except UnknownArchitecture as error:
		parser.error(str(error))


def summarize_command(args, parser):
	"""Print the pooled and per-domain retrieval table over every run file, and file it if asked.

	The table is generated from the run files on every invocation, so it can never drift from
	what the runs recorded the way a hand-written table would. Runs measured at different
	configurations are refused rather than blended, because such a row would compare setups
	instead of retrievers.
	"""
	from evaluation.summary import load_runs, render_markdown, summarize

	results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

	try:
		table = render_markdown(summarize(load_runs(results_dir)))
	except (FileNotFoundError, ValueError) as error:
		parser.error(str(error))

	print(table, end="")

	if args.out:
		out = Path(args.out)
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_text(table)
		print(f"[SUMMARY] wrote {out}")


def corpus_command(args, parser):
	"""Populate or check the corpus, and say plainly when a named domain is not in the manifest."""
	if args.command == "fetch-corpus":
		print("[CLI] Fetching the documents named by the corpus manifest...")
		operation, verb = populate, "fetched"
	else:
		print("[CLI] Verifying the documents on disk against the corpus manifest...")
		operation, verb = verify_corpus, "verified"

	try:
		outcomes = operation(load_manifest(), CORPUS_DIR, domains=args.domain)
	except ValueError as error:
		parser.error(str(error))

	report_outcomes(verb, outcomes)


def report_outcomes(verb: str, outcomes: list[Outcome]):
	"""Print one line per document and stop the run when any of them failed.

	A fetch that installs seven of eight documents and reports success would be worse than one
	that fails: ingestion would then read a corpus nobody has checked.
	"""
	for outcome in outcomes:
		status = str(outcome.action) if outcome.ok else f"{outcome.action} ({outcome.message})"
		print(f"[CORPUS] {outcome.document.id}: {status}")

	failed = [outcome for outcome in outcomes if not outcome.ok]
	print(f"[CORPUS] {len(outcomes) - len(failed)} of {len(outcomes)} documents {verb}")

	if failed:
		raise SystemExit(1)


def cli():
	"""Console entry point, so the harness runs without anyone having to know the module layout."""
	asyncio.run(main())


if __name__ == "__main__":
	cli()
