from basal_station.eval import evaluate, evaluate_question


def _choice_answer(choice, conf):
    return {"type": "choice", "choice": choice, "confidence": conf,
            "probabilities": {choice: conf}}


def test_choice_accuracy_and_confusion():
    pairs = [
        (_choice_answer("a", 0.9), "a"),
        (_choice_answer("b", 0.8), "a"),
        (_choice_answer("a", 0.7), "a"),
    ]
    report = evaluate_question("q", "choice", pairs)
    assert report["accuracy"] == round(2 / 3, 3)
    assert report["confusion"]["a"] == {"a": 2, "b": 1}
    assert report["n"] == 3


def test_noul_counts():
    pairs = [
        ({"type": "noul", "probabilities": {"true": 0.9}}, True),
        ({"type": "noul", "probabilities": {"true": 0.9}}, False),  # FP
        ({"type": "noul", "probabilities": {"true": 0.1}}, False),
        ({"type": "noul", "probabilities": {"true": 0.2}}, True),   # FN
    ]
    report = evaluate_question("q", "noul", pairs)
    assert report["accuracy"] == 0.5
    assert report["false_positives"] == 1
    assert report["false_negatives"] == 1


def test_score_mae():
    pairs = [
        ({"type": "score", "choice": "0.4"}, 0.5),
        ({"type": "score", "choice": "1.0"}, 0.5),
    ]
    report = evaluate_question("q", "score", pairs)
    assert report["mae"] == 0.3
    assert report["max_error"] == 0.5


def test_calibration_buckets():
    pairs = [
        (_choice_answer("a", 0.95), "a"),
        (_choice_answer("a", 0.55), "b"),
    ]
    report = evaluate_question("q", "choice", pairs)
    cal = report["calibration"]
    assert {b["confidence_range"] for b in cal} == {"0.5-0.6", "0.9-1.0"}
    top = next(b for b in cal if b["confidence_range"] == "0.9-1.0")
    assert top["observed_accuracy"] == 1.0


def test_evaluate_overall_and_missing_labels():
    questions = {"q": {"type": "choice"}, "skip": {"type": "noul"}}
    results = [{"answers": {"q": _choice_answer("a", 0.9)}}, {"answers": {}}]
    labels = [{"q": "a"}, {"q": "a"}]
    report = evaluate(questions, results, labels)
    assert report["questions"]["q"]["accuracy"] == 1.0
    assert report["questions"]["skip"]["n"] == 0
    assert report["overall"] == 1.0
