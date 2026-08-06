from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.ragas_compat import install as install_ragas_compat  # noqa: E402

install_ragas_compat()

from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.pipeline import FinancialGraphRAGPipeline  # noqa: E402

logger = logging.getLogger(__name__)

RAGAS_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

METRIC_NAMES: Dict[str, str] = {
    "faithfulness": "Faithfulness (Fidelidad)",
    "answer_relevancy": "Answer Relevancy (Relevancia de la Respuesta)",
    "context_precision": "Context Precision (Precisión del Contexto)",
    "context_recall": "Context Recall (Exhaustividad del Contexto)",
}


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    expected_ticker: str
    expected_year: int
    expected_section: str


@dataclass
class EvalResult:
    sample: EvalSample
    answer: str
    contexts: List[str]
    scores: Dict[str, float] = field(default_factory=dict)


def load_test_dataset(path: str | Path = "evals/test_dataset.json") -> List[EvalSample]:
    path = Path(path)
    if not path.exists():
        logger.warning("Test dataset not found at %s; using empty set", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = [
        EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
            expected_ticker=item.get("expected_ticker", ""),
            expected_year=item.get("expected_year", 0),
            expected_section=item.get("expected_section", ""),
        )
        for item in raw
    ]
    logger.info("Loaded %d evaluation samples from %s", len(samples), path)
    return samples


class RagasEvaluator:
    def __init__(
        self,
        pipeline: FinancialGraphRAGPipeline,
        llm: BaseChatModel,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        self.pipeline = pipeline
        self.llm = llm
        self.embeddings = embeddings

    def run(
        self,
        samples: List[EvalSample],
        output_path: Optional[str | Path] = None,
    ) -> List[EvalResult]:
        if not samples:
            logger.warning("No samples to evaluate")
            return []

        results: List[EvalResult] = []

        for idx, sample in enumerate(samples):
            logger.info(
                "Evaluating [%d/%d]: %s",
                idx + 1,
                len(samples),
                sample.question[:80],
            )
            try:
                retrieval_result = self.pipeline.query(sample.question)
                contexts = [
                    c.text for c in retrieval_result.final_context
                ]

                results.append(
                    EvalResult(
                        sample=sample,
                        answer=retrieval_result.answer,
                        contexts=contexts,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Pipeline failed for question '%s': %s",
                    sample.question[:80],
                    exc,
                )
                results.append(
                    EvalResult(
                        sample=sample,
                        answer="",
                        contexts=[],
                    )
                )

        scores = self._compute_ragas(results)

        valid_idx = [
            i for i, r in enumerate(results) if r.answer and r.contexts
        ]
        for idx, r in enumerate(results):
            if idx not in valid_idx:
                r.scores = {m: 0.0 for m in METRIC_NAMES}
                continue
            pos = valid_idx.index(idx)
            r.scores = {
                m: scores[m][pos] if scores[m] else 0.0
                for m in METRIC_NAMES
            }

        if output_path:
            self._report(results, output_path)

        return results

    def _compute_ragas(
        self,
        results: List[EvalResult],
    ) -> Dict[str, List[float]]:
        valid = [
            r for r in results if r.answer and r.contexts
        ]

        if not valid:
            logger.warning("No valid results to compute RAGAS metrics")
            return {m: [] for m in METRIC_NAMES}

        dataset = Dataset.from_dict({
            "question": [r.sample.question for r in valid],
            "answer": [r.answer for r in valid],
            "contexts": [r.contexts for r in valid],
            "ground_truth": [r.sample.ground_truth for r in valid],
        })

        try:
            result = evaluate(
                dataset=dataset,
                metrics=RAGAS_METRICS,
                llm=self.llm,
                embeddings=self.embeddings,
            )

            scores: Dict[str, List[float]] = {}
            for m in METRIC_NAMES:
                col = result[m]
                if hasattr(col, "to_list"):
                    scores[m] = [float(v) for v in col.to_list()]
                else:
                    scores[m] = [float(col)] * len(valid)

            return scores

        except Exception as exc:
            logger.error("RAGAS evaluation failed: %s", exc)
            return {m: [0.0] * len(valid) for m in METRIC_NAMES}

    def _report(
        self,
        results: List[EvalResult],
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report: List[dict] = []
        for r in results:
            report.append({
                "question": r.sample.question,
                "ground_truth": r.sample.ground_truth,
                "answer": r.answer,
                "context_count": len(r.contexts),
                "scores": r.scores,
                "expected_ticker": r.sample.expected_ticker,
                "expected_year": r.sample.expected_year,
                "expected_section": r.sample.expected_section,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        avg_scores: Dict[str, float] = {}
        for m in METRIC_NAMES:
            vals = [r.scores.get(m, 0.0) for r in results]
            avg_scores[m] = sum(vals) / len(vals) if vals else 0.0

        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_samples": len(results),
                    "average_scores": {
                        METRIC_NAMES.get(k, k): v
                        for k, v in avg_scores.items()
                    },
                    "individual_results": report,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Evaluation report saved to %s", summary_path)

        print("\n" + "=" * 60)
        print("RAGAS EVALUATION SUMMARY")
        print("=" * 60)
        for key, name in METRIC_NAMES.items():
            val = avg_scores.get(key, 0.0)
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {name:45s} {bar} {val:.4f}")
        print("=" * 60)


def main(
    llm: BaseChatModel,
    embeddings: Optional[Embeddings] = None,
    dataset_path: str | Path = "evals/test_dataset.json",
    output_path: str | Path = "evals/results/eval_report.json",
    skip_ingest: bool = False,
    tickers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    pipeline = FinancialGraphRAGPipeline(llm=llm)

    if tickers and years and not skip_ingest:
        for ticker in tickers:
            for year in years:
                logger.info("Ingesting %s / %s", ticker, year)
                pipeline.ingest_and_index(ticker, year)

    samples = load_test_dataset(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    evaluator = RagasEvaluator(pipeline=pipeline, llm=llm, embeddings=embeddings)
    evaluator.run(samples=samples, output_path=output_path)

    pipeline.close()


if __name__ == "__main__":
    import argparse

    from src.llm_factory import create_embeddings, create_llm

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the financial GraphRAG pipeline"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N samples",
    )
    parser.add_argument(
        "--dataset",
        default="evals/test_dataset.json",
        help="Path to the test dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="evals/results/eval_report.json",
        help="Path for the evaluation report JSON",
    )
    args = parser.parse_args()

    llm = create_llm()

    main(
        llm=llm,
        embeddings=create_embeddings(),
        dataset_path=args.dataset,
        output_path=args.output,
        max_samples=args.samples,
    )
