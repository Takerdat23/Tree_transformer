import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from data_utils.vocab import Vocab
from data_utils.dataset import PhoNERT_Dataset
from data_utils import collate_fn
from transformer import Transformer
from evaluation import F1

from tqdm import tqdm
import json
import argparse
import os

device = torch.device("mps")

def train(model: nn.Module, loss_fn, optimizer, dataloader: DataLoader, epoch: int):
    model.train()
    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for input_ids, tags in pbar:
            # forward
            input_ids = input_ids.to(device)
            tags = tags.to(device)
            _, logits = model(input_ids)
            # backward
            optimizer.zero_grad()
            loss = loss_fn(logits, tags)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "Loss": loss.item()
            })
            pbar.update()

def validate(model: nn.Module, dataloader: DataLoader, epoch: int):
    model.eval()
    f1_scorer = F1()
    scores = []
    with tqdm(dataloader, desc=f"Epoch {epoch} - Validating") as pbar:
        for input_ids, tags in pbar:
            input_ids = input_ids.to(device)
            tags = tags.to(device)
            outputs, _ = model(input_ids)
            score = f1_scorer.compute_score(tags, outputs)
            scores.append(score)

            pbar.set_postfix({
                "F1 score": sum(scores) / len(scores)
            })
            pbar.update()

    return sum(scores) / len(scores)

def evaluate(model: nn.Module, dataloader: DataLoader, vocab: Vocab):
    model.eval()
    f1_scorer = F1()
    scores = []
    results = {
        "predictions": [],
        "f1": 0
    }
    with tqdm(dataloader, desc=f"Evaluating") as pbar:
        for input_ids, tags in pbar:
            input_ids = input_ids.to(device)
            tags = tags.to(device)
            outputs, _ = model(input_ids)

            sentence = vocab.decode_sentence(input_ids)[0]
            predicted_tag = vocab.decode_tag(outputs)[0]
            gt_tag = vocab.decode_tag(tags)[0]
            results["predictions"].append({
                "sentence": sentence,
                "predicted_tags": predicted_tag,
                "gt_tags": gt_tag
            })

            score = f1_scorer.compute_score(tags, outputs)
            scores.append(score)

            pbar.set_postfix({
                "F1 score": sum(scores) / len(scores)
            })
            pbar.update()

    results["f1"] = sum(scores) / len(scores)

    return results

def save_checkpoint(model: nn.Module, checkpoint_path: str):
    pass

def start(args):
    print("Creating vocab ...")
    vocab = Vocab([
        args.train_path,
        args.dev_path,
        args.test_path,
    ])

    print("Creating dataset ...")
    train_dataset = PhoNERT_Dataset(
        args.train_path, 
        vocab
    )
    dev_dataset = PhoNERT_Dataset(
        args.dev_path,
        vocab
    )
    test_dataset = PhoNERT_Dataset(
        args.test_path, 
        vocab
    )

    print("Creating data loader ...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print("Creating model ...")
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heas,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        vocab=vocab
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.padding_idx).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_f1 = 0
    patient = 0
    epoch = 1
    while True:
        train(model, loss_fn, optimizer, train_dataloader, epoch)
        f1_score = validate(model, dev_dataloader, epoch)
        if f1_score > best_f1:
            best_f1 = f1_score
            save_checkpoint(model, "best_model.pth", args.checkpoint_path)
        else:
            save_checkpoint(model, "last_model.pth", args.checkpoint_path)
            patient += 1
            if patient > 5:
                break
    
    print("Evaluating on the test set ...")
    results = evaluate(model, test_dataloader, vocab)
    json.dump(results, os.path.join(
        args.checkpoint_path,
        "results.json"
    ), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    # arguments for the dataset
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--dev-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)

    args = parser.parse_args()

    start(args)

    print("Task finished.")
