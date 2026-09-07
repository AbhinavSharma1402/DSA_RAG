"""Evaluation script for the RAG system."""
import json
import sys
import logging
from typing import List, Dict, Any
from app.rag.pipeline import get_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG system performance."""
    
    def __init__(self, questions_file: str = "evaluation/questions.json"):
        """Initialize evaluator with test questions."""
        with open(questions_file) as f:
            self.questions = json.load(f)
        
        self.pipeline = get_rag_pipeline()
        self.results = []
    
    def evaluate_all(self, verbose: bool = False) -> Dict[str, Any]:
        """Run evaluation on all questions."""
        logger.info(f"Starting evaluation on {len(self.questions)} questions...")
        
        for i, question_data in enumerate(self.questions, 1):
            query = question_data.get("q")
            
            if query.startswith("_"):
                logger.info(f"[{i}/{len(self.questions)}] Skipping: {query}")
                continue
            
            logger.info(f"[{i}/{len(self.questions)}] Evaluating: {query}")
            
            result = self.evaluate_single(question_data, verbose=verbose)
            self.results.append(result)
            
            if verbose:
                self.print_result(result)
        
        return self.compute_metrics()
    
    def evaluate_single(self, question_data: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """Evaluate a single question."""
        query = question_data.get("q")
        expect_refusal = question_data.get("expect_refusal", False)
        expect_video_id = question_data.get("expect_video_id")
        expect_around_sec = question_data.get("expect_around_sec")
        tolerance_sec = question_data.get("tolerance_sec", 120)
        category = question_data.get("category", "unknown")
        
        try:
            # Get response from pipeline
            response = self.pipeline.process_query(
                query=query,
                conversation_id=None,
                topic=None,
                mode="lecture",
                debug=verbose
            )
            
            # Evaluate based on expectation
            if expect_refusal:
                # Check if system refused to answer
                hit = self.is_refusal(response)
                result_type = "refusal"
            else:
                # Check if retrieval is correct
                hit = self.check_retrieval(
                    response,
                    expect_video_id,
                    expect_around_sec,
                    tolerance_sec
                )
                result_type = "retrieval"
            
            return {
                "query": query,
                "category": category,
                "result_type": result_type,
                "hit": hit,
                "confidence": response.confidence,
                "retrieved_chunks": response.retrieved_chunks,
                "rewritten_query": response.rewritten_query,
                "sources_count": len(response.sources),
                "answer_preview": response.answer[:100],
                "full_response": response if verbose else None
            }
        
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            return {
                "query": query,
                "category": category,
                "result_type": "error",
                "hit": False,
                "error": str(e)
            }
    
    def is_refusal(self, response) -> bool:
        """Check if response is a refusal."""
        refusal_indicators = [
            "couldn't find",
            "not found",
            "no information",
            "insufficient",
            "can't answer"
        ]
        
        answer_lower = response.answer.lower()
        return any(indicator in answer_lower for indicator in refusal_indicators)
    
    def check_retrieval(self, response, expect_video_id: str,
                       expect_around_sec: int, tolerance_sec: int) -> bool:
        """Check if retrieval contains expected content."""
        if not response.sources:
            return False
        
        # Check if any source matches
        for source in response.sources:
            if source.video_id == expect_video_id:
                # Check timestamp
                time_diff = abs(source.timestamp - expect_around_sec)
                if time_diff <= tolerance_sec:
                    return True
        
        return False
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if not self.results:
            return {"error": "No results to compute metrics"}
        
        retrieval_results = [r for r in self.results if r["result_type"] == "retrieval"]
        refusal_results = [r for r in self.results if r["result_type"] == "refusal"]
        error_results = [r for r in self.results if r["result_type"] == "error"]
        
        metrics = {
            "total_questions": len(self.results),
            "total_hits": sum(1 for r in self.results if r.get("hit")),
            "overall_accuracy": sum(1 for r in self.results if r.get("hit")) / len(self.results),
            
            "retrieval": {
                "total": len(retrieval_results),
                "hits": sum(1 for r in retrieval_results if r.get("hit")),
                "accuracy": sum(1 for r in retrieval_results if r.get("hit")) / len(retrieval_results) if retrieval_results else 0
            },
            
            "refusal": {
                "total": len(refusal_results),
                "hits": sum(1 for r in refusal_results if r.get("hit")),
                "accuracy": sum(1 for r in refusal_results if r.get("hit")) / len(refusal_results) if refusal_results else 0
            },
            
            "errors": len(error_results),
            
            "confidence_distribution": {
                "high": sum(1 for r in self.results if r.get("confidence") == "high"),
                "medium": sum(1 for r in self.results if r.get("confidence") == "medium"),
                "low": sum(1 for r in self.results if r.get("confidence") == "low")
            }
        }
        
        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "hits": 0}
            categories[cat]["total"] += 1
            if result.get("hit"):
                categories[cat]["hits"] += 1
        
        metrics["by_category"] = categories
        
        return metrics
    
    def print_result(self, result: Dict[str, Any]):
        """Print a single result."""
        status = "✓ PASS" if result.get("hit") else "✗ FAIL"
        query = result.get("query", "")[:60]
        
        print(f"{status} | {query}")
        print(f"   Category: {result.get('category')} | "
              f"Confidence: {result.get('confidence')} | "
              f"Sources: {result.get('sources_count')}")
        
        if result.get("rewritten_query"):
            print(f"   Rewritten: {result['rewritten_query'][:60]}")
        print()
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.1%}")
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Total Hits: {metrics['total_hits']}")
        print(f"Total Errors: {metrics['errors']}")
        
        print(f"\nRetrieval Questions: {metrics['retrieval']['total']}")
        print(f"  Accuracy: {metrics['retrieval']['accuracy']:.1%}")
        
        print(f"\nRefusal Questions: {metrics['refusal']['total']}")
        print(f"  Accuracy: {metrics['refusal']['accuracy']:.1%}")
        
        print(f"\nConfidence Distribution:")
        for level, count in metrics['confidence_distribution'].items():
            print(f"  {level.upper()}: {count}")
        
        print(f"\nBy Category:")
        for cat, stats in metrics['by_category'].items():
            acc = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {cat}: {acc:.1%} ({stats['hits']}/{stats['total']})")
        
        print("\n" + "="*60 + "\n")
    
    def save_results(self, output_file: str = "evaluation/results.json"):
        """Save evaluation results to file."""
        data = {
            "results": self.results,
            "metrics": self.compute_metrics()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate YouTube DSA RAG system")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--questions", "-q", default="evaluation/questions.json",
                       help="Questions file path")
    parser.add_argument("--output", "-o", default="evaluation/results.json",
                       help="Output results file")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = RAGEvaluator(questions_file=args.questions)
    metrics = evaluator.evaluate_all(verbose=args.verbose)
    
    # Print summary
    evaluator.print_summary(metrics)
    
    # Save results
    evaluator.save_results(output_file=args.output)


if __name__ == "__main__":
    main()
