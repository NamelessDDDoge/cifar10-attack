import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_utils import iter_todo_batches, log_progress_event


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


def test_iter_todo_batches_does_not_overlap():
    todo = list(enumerate([f"{i}.png" for i in range(23)]))
    batches = list(iter_todo_batches(todo, batch_size=5))
    flattened = [item for batch in batches for item in batch]

    assert flattened == todo
    assert [len(batch) for batch in batches] == [5, 5, 5, 5, 3]


def test_log_progress_event_writes_jsonl(tmp_path):
    logger = _Logger()
    progress_path = tmp_path / "progress.jsonl"

    log_progress_event(
        logger=logger,
        tag="TEST",
        adv_dir=tmp_path / "adv_test",
        processed=10,
        total=500,
        progress_path=progress_path,
        batch=[0, 9],
    )

    rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert rows[0]["tag"] == "TEST"
    assert rows[0]["processed"] == 10
    assert rows[0]["total"] == 500
    assert rows[0]["batch"] == [0, 9]
    assert rows[0]["percent"] == 2.0
    assert logger.messages


if __name__ == "__main__":
    test_iter_todo_batches_does_not_overlap()
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        test_log_progress_event_writes_jsonl(Path(d))
    print("run_utils progress tests passed")
