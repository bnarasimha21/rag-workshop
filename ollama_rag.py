"""Original workshop entry point, kept so `python ollama_rag.py` still works.

The pipeline now lives in the `rag/` package; this file just starts the UI.
"""

from rag.app import main

if __name__ == "__main__":
    main()
