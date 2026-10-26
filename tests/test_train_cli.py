"""CLI forwarding: user-specified --imgsz must reach train_model (review item 1)."""
import src.models.train as train_mod


def test_cli_forwards_imgsz_and_training_args(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_mod, "train_model", lambda **kw: captured.update(kw))
    train_mod.main(["--imgsz", "320", "--epochs", "2", "--batch", "4"])
    assert captured["imgsz"] == 320
    assert captured["epochs"] == 2
    assert captured["batch"] == 4
