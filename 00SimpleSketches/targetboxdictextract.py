import torch

# Angenommen, targets ist eine Liste von Dictionaries
targets = [
    {
        "boxes": torch.tensor([10, 10, 20, 20], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    },
    {
        "boxes": torch.tensor([30, 30, 40, 40], dtype=torch.float32),
        "labels": torch.tensor([2], dtype=torch.int64),
    }
]

# Boxen aus den Targets extrahieren
boxes = torch.stack([item['boxes'] for item in targets])  # Jetzt sollte es funktionieren
print("targets", targets)
print("boxes", boxes)
