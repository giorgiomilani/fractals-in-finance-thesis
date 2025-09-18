import torch, torch.nn as nn
from torch.utils.data import DataLoader
from fractalfinance.gaf.dataset import GAFWindowDataset
import numpy as np, typer, json, pathlib, rich

app = typer.Typer()

@app.command()
def train(
    data_path: str = typer.Option(...),
    epochs: int = 5,
    batch: int = 32,
    lr: float = 1e-3,
    outdir: str = "experiments/outputs/cnn",
):
    series = np.load(data_path)        # assume .npy of returns
    ds = GAFWindowDataset(series, labels=None)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, 1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * 30 * 30, 128),
        nn.ReLU(), nn.Linear(128, 3)      # 3 synthetic regimes
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    model.train()
    history = []
    for epoch in range(epochs):
        loss_epoch = 0.0
        correct = 0
        total = 0
        for x, y in dl:
            y = y.long()
            pred = model(x)
            loss = crit(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch = x.size(0)
            loss_epoch += loss.item() * batch
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += batch
        avg_loss = loss_epoch / total if total else 0.0
        acc = correct / total if total else 0.0
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
        rich.print(f"[cyan]{epoch=} loss={avg_loss:.4f} acc={acc:.3f}")

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/cnn.pth")
    metrics = {
        "epochs": epochs,
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
        "final_accuracy": history[-1]["accuracy"] if history else None,
    }
    json.dump(metrics, open(f"{outdir}/metrics.json", "w"))

if __name__ == "__main__":
    app()
