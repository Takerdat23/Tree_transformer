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

def train(model: nn.Module, 
          loss_fn, optimizer, 
          dataloader: DataLoader, 
          epoch: int,
          device: str):
    model.train()
    losses = []
    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for input_ids, tags in pbar:
            # forward
            input_ids = input_ids.to(device)
            tags = tags.to(device)
            _, logits = model(input_ids)
            # backward
            optimizer.zero_grad()
            total_tags = logits.shape[-1]
            loss = loss_fn(
                logits.reshape(-1, total_tags),
                tags.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            pbar.set_postfix({
                "Loss": sum(losses) / len(losses)
            })
            pbar.update()

def validate(model: nn.Module, 
             dataloader: DataLoader, 
             epoch: int,
             device: str):
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

def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             vocab: Vocab,
             device: str):
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

def save_checkpoint(model: nn.Module, 
                    optimizer,
                    checkpoint_path: str):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict()
    }, checkpoint_path)

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
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        vocab=vocab
    ).to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.padding_idx).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_f1 = 0
    patient = 0
    epoch = 1
    while True:
        train(model, loss_fn, optimizer, train_dataloader, epoch, args.device)
        f1_score = validate(model, dev_dataloader, epoch, args.device)
        if f1_score > best_f1:
            best_f1 = f1_score
            save_checkpoint(model, 
                            optimizer, 
                            os.path.join(args.checkpoint_path, "best_model"))
        else:
            save_checkpoint(model, 
                            optimizer, 
                            os.path.join(args.checkpoint_path, "last_model"))
            patient += 1
            if patient > 5:
                break
    
    print("Evaluating on the test set ...")
    results = evaluate(model, test_dataloader, vocab, args.device)
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
    parser.add_argument("--num-workers", type=int, required=True)
    # arguments for the dataset
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--dev-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)

    args = parser.parse_args()

    start(args)

    print("Task finished.")
