import asyncio
import argparse
from pathlib import Path
from vector.store import reset_vector_store
from build_index import build_index
from runners.hybrid_rag import run_hybrid
from runners.naive_rag import run_naive
from corpus.manifest import load_manifest
from corpus.populate import Outcome, populate, verify_corpus

# Where the corpus is populated. It is the same directory ingestion walks, so a document needs
# to be in no other place.
CORPUS_DIR = Path("documents")


async def main():
	"""
	Main CLI parser and router for the RAG Comparison Benchmarking application.
	Allows users to ingest documents, clear the vector store, test models, or compare them.
	"""

	# create main argument parser
	parser = argparse.ArgumentParser(
		description="RAG Comparison Benchmarking CLI - A tool to test and compare RAG architectures."
	)
	
	# add subparsers for each command
	subparsers = parser.add_subparsers(dest="command", required=True, help="RAG commands")
	
	# ingest command
	ingest_parser = subparsers.add_parser(
		"ingest", 
		help="Reset vector store, load files from ./documents, split them, embed, and index into Qdrant"
	)
	ingest_parser.add_argument("architecture", type=str, help="The architecture for which we do the ingestion")

	
	# clear command
	clear_parser = subparsers.add_parser(
		"clear", 
		help="Clear/delete the Qdrant vector store collection to prevent data contamination"
	)
	clear_parser.add_argument("collection_name", type=str, help="The collection name to clear")
	
	# test-naive command
	naive_parser = subparsers.add_parser(
		"test-naive", 
		help="Retrieve context and answer a query using Naive RAG (Dense vector search only)"
	)
	naive_parser.add_argument("query", type=str, help="The query/question to run")
	
	# test-hybrid command
	hybrid_parser = subparsers.add_parser(
		"test-hybrid", 
		help="Retrieve context and answer a query using Hybrid RAG (Dense + BM42 Sparse Search + RRF)"
	)
	hybrid_parser.add_argument("query", type=str, help="The query/question to run")
	
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
	
	# route command to appropriate action
	if args.command == "ingest":
		print("[CLI] Ingesting documents...")
		
		archi = args.architecture
		if archi == "naive":
			build_index("rag_naive", enable_hybrid=False)
		elif archi == "hybrid":
			build_index("rag_hybrid", enable_hybrid=True)
		else:
			print("[INFO] Invalid architecture specified")

	elif args.command == "clear":
		print("[CLI] Clearing vector store...")
		
		reset_vector_store(args.collection_name)

	elif args.command == "test-naive":
		print("[CLI] Running NaiveRAG...")

		await run_naive(args.query)

	elif args.command == "test-hybrid":
		print("[CLI] Running HybridRAG...")
		
		await run_hybrid(args.query)

	elif args.command in ("fetch-corpus", "verify-corpus"):
		corpus_command(args, parser)


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


if __name__ == "__main__":
	asyncio.run(main())