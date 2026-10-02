import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neural_models import SentenceCNN,BiLSTM


class PaddingTests(unittest.TestCase):
    def test_padding_does_not_change_predictions(self):
        torch.manual_seed(1)
        lengths=torch.tensor([6,3])
        short=torch.tensor([[2,3,4,5,6,7,0,0],[2,3,4,0,0,0,0,0]])
        long=torch.nn.functional.pad(short,(0,7))
        for model in [SentenceCNN(10),BiLSTM(10)]:
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(short,lengths),model(long,lengths),atol=1e-6,rtol=1e-5)

    def test_models_train_with_short_prompts(self):
        ids=torch.tensor([[2,0,0,0,0],[3,4,5,0,0]])
        lengths=torch.tensor([1,3])
        for model in [SentenceCNN(10),BiLSTM(10)]:
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(ids,lengths),torch.tensor([0.,1.]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__=='__main__': unittest.main()
