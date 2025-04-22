#!/usr/bin/env python3
import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Import your KAN code
from CP_KAN import FixedKAN, FixedKANConfig

def main():
    parser = argparse.ArgumentParser("Run CP-KAN on ImageNet with optional CNN backbone + QUBO subset.")
    parser.add_argument("--data_dir", required=True,
                        help="Path to ImageNet (or any other dataset in ImageFolder style).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--complexity_weight", type=float, default=0.0)
    parser.add_argument("--max_degree", type=int, default=5)
    parser.add_argument("--trainable_coeffs", action="store_true")
    parser.add_argument("--no_qubo_hidden", action="store_true",
                        help="Skip QUBO for hidden layers.")
    parser.add_argument("--default_hidden_degree", type=int, default=2)
    parser.add_argument("--qubo_subset_size", type=int, default=2000,
                        help="Size of random subset for QUBO. Use 0 to disable.")
    parser.add_argument("--backbone_name", type=str, default="resnet18",
                        help="Which CNN backbone to use, or 'none' to disable.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on device: {device}")

    # ============ Dataset / Dataloaders ============
    # We'll use the torchvision.datasets.ImageNet class if available,
    # or we can fallback to ImageFolder if that doesn't work in your PyTorch version.
    # We'll assume you have "train" & "val" folders inside data_dir.

    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    try:
        train_dataset = torchvision.datasets.ImageNet(root=args.data_dir, split="train", transform=train_tf)
        val_dataset   = torchvision.datasets.ImageNet(root=args.data_dir, split="val", transform=val_tf)
    except:
        print("[WARN] torchvision.datasets.ImageNet not working, falling back to ImageFolder:")
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
        val_dataset   = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"[INFO] # Classes: {num_classes}")
    print(f"[INFO] # Train images: {len(train_dataset)}, # Val images: {len(val_dataset)}")

    # ============ Optional CNN Backbone ============
    if args.backbone_name.lower() == "none":
        print("[INFO] No CNN backbone => flatten images directly (3x224x224 => 150528-dim).")
        # We'll define a simple "feature_extractor" that just flattens:
        def extract_features(images: torch.Tensor) -> torch.Tensor:
            # images shape: [B,3,224,224]
            B = images.size(0)
            return images.view(B, -1)  # [B, 3*224*224 = 150528]
        feat_dim = 3*224*224

    else:
        print(f"[INFO] Using backbone: {args.backbone_name}")
        backbone_func = getattr(torchvision.models, args.backbone_name, None)
        if backbone_func is None:
            raise ValueError(f"Unknown backbone {args.backbone_name} in torchvision.models.")
        backbone = backbone_func(pretrained=True).to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        # For ResNet, e.g. remove last FC
        if "resnet" in args.backbone_name:
            feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
            # typical final feat dim
            if args.backbone_name in ["resnet18", "resnet34"]:
                feat_dim = 512
            else:
                feat_dim = 2048
        else:
            # Example for mobilenet_v2
            if args.backbone_name.startswith("mobilenet"):
                feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
                feat_dim = 1280
            else:
                raise NotImplementedError(f"Backbone {args.backbone_name} not yet handled.")

        def extract_features(images: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats = feature_extractor(images)
                # For ResNets => [B, feat_dim, 1, 1], flatten
                feats = feats.view(feats.size(0), -1)
            return feats

    # ============ Build KAN ============
    # We'll do something like [feat_dim -> feat_dim//2 -> num_classes], or pick your own shape
    hidden_dim = max(feat_dim // 2, 64)  # just to avoid zero
    network_shape = [feat_dim, hidden_dim, num_classes]
    config = FixedKANConfig(
        network_shape=network_shape,
        max_degree=args.max_degree,
        complexity_weight=args.complexity_weight,
        trainable_coefficients=args.trainable_coeffs,
        skip_qubo_for_hidden=args.no_qubo_hidden,
        default_hidden_degree=args.default_hidden_degree
    )
    kan_model = FixedKAN(config).to(device)

    # ============ QUBO subset ============
    # We'll gather a small random subset from the training set, extract features, do QUBO
    if args.qubo_subset_size > 0:
        print(f"[INFO] QUBO subset size: {args.qubo_subset_size}")
        subset_indices = random.sample(range(len(train_dataset)), args.qubo_subset_size)
        qubo_subset = Subset(train_dataset, subset_indices)
        qubo_loader = DataLoader(qubo_subset, batch_size=32, shuffle=False, num_workers=2)

        all_feats = []
        all_labels = []
        for imgs, lbls in qubo_loader:
            imgs = imgs.to(device)
            feats = extract_features(imgs)
            all_feats.append(feats.cpu())
            all_labels.append(lbls.cpu())
        all_feats = torch.cat(all_feats, dim=0)
        all_labels_int = torch.cat(all_labels, dim=0)
        onehot_labels = F.one_hot(all_labels_int, num_classes=num_classes).float()

        print("[INFO] Running QUBO-based selection on subset...")
        # We'll call train_model_cross_entropy w/ 0 epochs => only QUBO
        kan_model.train_model_cross_entropy(
            x_data=all_feats.to(device),
            y_data_int=all_labels_int.to(device),
            y_data_onehot=onehot_labels.to(device),
            num_epochs=0,
            lr=args.lr,
            complexity_weight=args.complexity_weight,
            do_qubo=True
        )
        print("[INFO] QUBO done. Polynomial degrees set.")
    else:
        print("[INFO] Skipping QUBO step (subset_size=0) => all degrees remain default or uninitialized if skip_qubo_for_hidden=False!")

    # ============ Final Training Loop ============
    # We'll do a standard training loop
    params_to_train = []
    for layer in kan_model.layers:
        params_to_train.extend([layer.combine_W, layer.combine_b])
        for nrn in layer.neurons:
            params_to_train.extend([nrn.w, nrn.b])
            if config.trainable_coefficients and nrn.coefficients is not None:
                params_to_train.extend(list(nrn.coefficients))

    optimizer = torch.optim.Adam(params_to_train, lr=args.lr)

    def train_one_epoch(epoch):
        kan_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start_time = time.time()

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            feats = extract_features(imgs)
            logits = kan_model(feats)
            ce = F.cross_entropy(logits, lbls)
            w_norm = 0.0
            loss = ce + config.complexity_weight*w_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(lbls).sum().item()
            total_samples += imgs.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} [Train] Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={elapsed:.1f}s")

    def validate():
        kan_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                feats = extract_features(imgs)
                logits = kan_model(feats)
                ce = F.cross_entropy(logits, lbls)
                total_loss += ce.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                total_correct += preds.eq(lbls).sum().item()
                total_samples += imgs.size(0)
        avg_loss = total_loss / total_samples
        acc = 100.0 * total_correct / total_samples
        return avg_loss, acc

    print(f"[INFO] Starting final training for {args.epochs} epochs...")
    for e in range(1, args.epochs+1):
        train_one_epoch(e)
        val_loss, val_acc = validate()
        print(f"Epoch {e} [Val] Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

    print("[DONE] Training complete!")


if __name__ == "__main__":
    main()