"""Offline evaluation harness for the RAG pipeline.

Measures retrieval quality (nDCG, MRR, Recall), answer faithfulness against the
retrieved context, and attributes each failure to retrieval or generation.
"""
