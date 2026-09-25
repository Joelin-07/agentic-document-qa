"""Run the evaluation dataset and print/save the metrics.

    python scripts/run_eval.py              # all questions
    python scripts/run_eval.py --judge      # + LLM-as-judge scoring
    python scripts/run_eval.py --limit 5    # quick check

Same as `docqa eval`. Reports are written to data/eval_reports/ (JSON + Markdown).
"""

import sys

from docqa.cli import app

if __name__ == "__main__":
    sys.exit(app(["eval", *sys.argv[1:]]))
