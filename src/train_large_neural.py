"""Train CNN/BiLSTM or fully fine-tune DistilBERT on the shared 100k splits."""
import argparse
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch import nn
from evaluation import ROOT, metrics, code_hashes
from large_data import PARTITIONS, rows, validate_manifest, encode_corpus
from neural_models import SentenceCNN, BiLSTM
from train_neural import setup_seed, synchronize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=['cnn', 'bilstm', 'distilbert'])
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed/large')
    p.add_argument('--output', type=Path, default=ROOT / 'results/large')
    p.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/large')
    p.add_argument('--encoder', type=Path, default=ROOT / 'data/models/distilbert')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--seed', type=int, default=4442)
    p.add_argument('--benchmark-steps', type=int, default=0)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation) < 1 or args.max_tokens < 5:
        p.error('Invalid training dimensions')
    setup_seed(args.seed)
    device = torch.device(args.device)
    manifest = validate_manifest(args.data)
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    artifact = args.artifacts / args.model; artifact.mkdir(parents=True, exist_ok=True)
    transformer = args.model == 'distilbert'
    start = time.perf_counter()
    encoded, vocab = encode_corpus(args.data, artifact / 'tokens', manifest, args.max_tokens, transformer, args.encoder)
    feature_seconds = time.perf_counter() - start
    if transformer:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(args.encoder, num_labels=2, local_files_only=True)
        model.requires_grad_(True)
        # Checkpointing trades compute for VRAM; all encoder layers receive gradients.
        model.gradient_checkpointing_enable()
        lr = 2e-5
        encoder_initial = model.distilbert.transformer.layer[0].attention.q_lin.weight.detach().clone()
    else:
        model = SentenceCNN(len(vocab)+2) if args.model == 'cnn' else BiLSTM(len(vocab)+2)
        lr = .001
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01 if transformer else 0)
    n = manifest['partitions']['train']['rows']
    steps_per_epoch = math.ceil(math.ceil(n / args.batch_size) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * .1) if transformer else 0
    def schedule(step):
        if not transformer:
            return 1.
        if step < warmup:
            return (step+1) / max(1, warmup)
        return max(0., (total_steps-step) / max(1, total_steps-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp = device.type == 'cuda' and transformer
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    def batch_values(split, ix):
        ids, lengths, labels = encoded[split]
        return (torch.from_numpy(np.asarray(ids[ix], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(lengths[ix], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(labels[ix], dtype=np.float32)).to(device))
    def forward(ids, lengths, labels=None):
        if transformer:
            mask = torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]
            result = model(input_ids=ids, attention_mask=mask, labels=labels.long() if labels is not None else None)
            return result.loss if labels is not None else result.logits.softmax(-1)[:, 1]
        logits = model(ids, lengths)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels) if labels is not None else logits.sigmoid()
    def predict(split):
        model.eval(); preds = []
        with torch.inference_mode():
            for offset in range(0, len(encoded[split][0]), args.batch_size):
                ids, lengths, _ = batch_values(split, slice(offset, offset+args.batch_size))
                with torch.autocast(device.type, enabled=amp):
                    preds.append(forward(ids, lengths).float().cpu().numpy())
        return np.concatenate(preds)
    history = []; best = -1.; optimizer_steps = 0; gradient_verified = False
    synchronize(device); started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); order = np.random.default_rng(args.seed+epoch).permutation(n)
        total_loss = 0.; optimizer.zero_grad(set_to_none=True)
        minibatches = math.ceil(n / args.batch_size)
        for step, offset in enumerate(range(0, n, args.batch_size)):
            ix = order[offset:offset+args.batch_size]
            ids, lengths, labels = batch_values('train', ix)
            # Correct scaling for the last, possibly partial accumulation window.
            window = min(args.accumulation, minibatches - (step // args.accumulation)*args.accumulation)
            with torch.autocast(device.type, enabled=amp):
                loss = forward(ids, lengths, labels)
            scaler.scale(loss/window).backward()
            total_loss += loss.item() * len(ix)
            if (step+1) % args.accumulation == 0 or step+1 == minibatches:
                scaler.unscale_(optimizer)
                if transformer and not gradient_verified:
                    gradients = [layer.attention.q_lin.weight.grad for layer in model.distilbert.transformer.layer]
                    if not all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients):
                        raise RuntimeError('Not all DistilBERT encoder layers received nonzero gradients')
                    gradient_verified = True
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
            if step % 200 == 0:
                print(f'{args.model} epoch {epoch+1}/{args.epochs}, rows {offset+len(ix)}/{n}, loss {loss.item():.4f}', flush=True)
            if args.benchmark_steps and step+1 == args.benchmark_steps:
                synchronize(device); elapsed = time.perf_counter()-started
                report = {'model': args.model, 'benchmark_only': True, 'rows_processed': offset+len(ix),
                          'seconds': elapsed, 'estimated_epoch_seconds': elapsed*n/(offset+len(ix)),
                          'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/2**20,
                          'encoder_gradient_verified': gradient_verified, 'config': vars(args) | {k:str(v) for k,v in vars(args).items() if isinstance(v,Path)}}
                (output / f'{args.model}_benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(report, indent=2)); return
        val = predict('val')
        val_rows = [{'label': int(y)} for y in encoded['val'][2]]
        score = metrics(val_rows, val)
        history.append({'epoch': epoch+1, 'train_loss': total_loss/n, 'val': score})
        print(f'{args.model} epoch {epoch+1} validation F1 {score["f1"]:.4f}', flush=True)
        if score['f1'] > best:
            best = score['f1']; chosen_epoch = epoch+1
            torch.save(model.state_dict(), artifact / 'best.pt')
    synchronize(device); training_seconds = time.perf_counter()-started
    encoder_changed = bool(transformer and not torch.equal(encoder_initial, model.distilbert.transformer.layer[0].attention.q_lin.weight.detach().cpu()))
    model.load_state_dict(torch.load(artifact / 'best.pt', map_location=device, weights_only=True))
    scores = {}; predictions = {}
    for split in PARTITIONS:
        preds = predict(split); predictions[split] = preds
        metadata = [{k:r[k] for k in ('id','label','family','group')} for r in rows(args.data / f'{split}.jsonl')]
        scores[split] = metrics(metadata, preds)
        for family in sorted(set(r['family'] for r in metadata)):
            indices = [i for i,r in enumerate(metadata) if r['family']==family]
            scores[f'{split}/{family}'] = metrics([metadata[i] for i in indices], preds[indices])
        np.savez_compressed(artifact / f'{split}_predictions.npz',
                            ids=np.array([r['id'] for r in metadata]), labels=encoded[split][2], probabilities=preds)
    if transformer:
        if not gradient_verified or not encoder_changed:
            raise RuntimeError('Full encoder fine-tuning was not verified')
        model.save_pretrained(artifact / 'model')
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.encoder, local_files_only=True).save_pretrained(artifact / 'model')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    result = {'model': args.model, 'config': config, 'train_rows': n, 'epochs_run': args.epochs,
              'chosen_epoch': chosen_epoch, 'selection': 'highest validation F1; fixed threshold 0.5',
              'learning_rate': lr, 'encoder_frozen': False if transformer else None,
              'encoder_gradient_verified': gradient_verified if transformer else None,
              'encoder_weight_changed': encoder_changed if transformer else None,
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'optimizer_steps': optimizer_steps, 'feature_seconds': feature_seconds,
              'training_and_validation_seconds': training_seconds, 'history': history, 'metrics': scores,
              'data_hashes': {s:manifest['partitions'][s]['sha256'] for s in PARTITIONS},
              'code_hashes': code_hashes(['train_large_neural.py','large_data.py','neural_models.py','evaluation.py']),
              'model_source': json.loads((args.encoder / 'source.json').read_text()) if transformer else None,
              'environment': {'torch':torch.__version__, 'python':platform.python_version(),
                              'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None},
              'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
    (output / f'{args.model}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'model':args.model, 'test_f1':scores['test']['f1'],
                      'official_eval_f1':scores['official_eval']['f1'], 'seconds':training_seconds}, indent=2))


if __name__ == '__main__':
    main()
