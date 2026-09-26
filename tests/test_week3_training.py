"""Check that feature caching and microbatches preserve the training objective."""
import unittest
import copy
import torch
from week3 import FastflowModel, FastflowLoss, feature_loss, train_epoch


class TrainingTests(unittest.TestCase):
    def test_cached_features_and_accumulated_gradients(self):
        torch.manual_seed(42)
        model = FastflowModel((32, 32), 'resnet18', pre_trained=False,
                              flow_steps=2, conv3x3_only=True)
        model.train()
        model.feature_extractor.eval()
        images = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            features = model.feature_extractor(images)
        direct = FastflowLoss()(*model(images))
        cached = feature_loss(model, features)
        torch.testing.assert_close(cached, direct)
        cached.backward()
        expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        for offset in [0, 2]:
            loss = feature_loss(model, [f[offset:offset + 2] for f in features]) / 2
            loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad, expected[name], rtol=1e-3, atol=1e-3)
        self.assertTrue(all(p.grad is None for p in model.feature_extractor.parameters()))

    def test_optimizer_and_shuffle_resume(self):
        torch.manual_seed(7)
        model = FastflowModel((32, 32), 'resnet18', pre_trained=False,
                              flow_steps=2, conv3x3_only=True)
        model.feature_extractor.eval()
        with torch.no_grad():
            features = model.feature_extractor(torch.randn(8, 3, 32, 32))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
        generator = torch.Generator().manual_seed(7)
        config = {'batch_size': 4, 'micro_batch_size': 2}
        train_epoch(model, features, optimizer, generator, config, 'cpu')
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        shuffle_state = generator.get_state().clone()
        expected_loss = train_epoch(model, features, optimizer, generator, config, 'cpu')
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        generator.set_state(shuffle_state)
        resumed_loss = train_epoch(model, features, optimizer, generator, config, 'cpu')
        self.assertEqual(expected_loss, resumed_loss)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
