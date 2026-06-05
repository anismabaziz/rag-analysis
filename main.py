import asyncio
import argparse     
from vector.store import reset_vector_store
from build_index import build_index
from runners.hybrid_rag import run_hybrid
from runners.naive_rag import run_naive


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
		help="Retrieve context and answer a query using Hybrid RAG (Dense + BM25 Sparse Search + RRF)"
	)
	hybrid_parser.add_argument("query", type=str, help="The query/question to run")
	
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


if __name__ == "__main__":
	asyncio.run(main())