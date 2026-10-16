"""Checks for the paired feature-fusion experiment."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings
from scripts.week5_model import ProjectedFlow, Projection, fit_projection, seed_all, train_epoch, torch, np


class FakeDataset(SimpleNamespace):
    def __len__(self):
        return len(self.paths)


class FusionTests(unittest.TestCase):
    def test_projection_matches_matrix_calculation_and_feature_sources(self):
        seed_all(1)
        saved = {'levels': []}
        features = []
        for c,h,k in [(40,8,32),(100,4,96),(40,2,32)]:
            saved['levels'].append({'components':torch.randn(k,c), 'mean':torch.randn(c),
                'offset':torch.randn(k),'scale':torch.rand(k)+.1})
            features.append(torch.randn(2,c,h,h,dtype=torch.float64))
        saved['levels']=[{key:value.double() for key,value in level.items()} for level in saved['levels']]
        projection=Projection(saved,8)
        actual=projection(features,'middle')
        p=saved['levels'][1]
        expected=(features[1].permute(0,2,3,1)-p['mean']) @ p['components'].T
        expected=((expected-p['offset'])/p['scale']).permute(0,3,1,2)
        expected=torch.nn.functional.interpolate(expected,(8,8),mode='bilinear',align_corners=False)
        torch.testing.assert_close(actual,expected)
        fusion=projection(features,'fusion')
        torch.testing.assert_close(fusion[:,32:64],actual[:,:32])
        changed=[x.clone() for x in features]
        changed[0]+=1
        torch.testing.assert_close(projection(changed,'middle'),actual,rtol=0,atol=0)
        self.assertFalse(torch.equal(projection(changed,'fusion')[:,:32],fusion[:,:32]))
        self.assertEqual(fusion.shape,(2,96,8,8))

    def test_projection_rejects_test_images(self):
        dataset=SimpleNamespace(paths=[Path('bottle/test/good/000.png')])
        with self.assertRaises(ValueError):
            fit_projection([],dataset,{})

    def test_constant_training_features_stay_finite(self):
        dataset=FakeDataset(root=Path('bottle'),paths=[Path(f'bottle/train/good/{i}.png') for i in range(2)])
        sources=[SimpleNamespace(values=np.zeros((2,100,8,8),dtype=np.float32)) for _ in range(3)]
        config={'projection_seed':2026,'pca_sample_limit':128,'pca_solver':'randomized',
                'pca_power_iterations':1,'scale_floor':1e-6}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            saved=fit_projection(sources,dataset,config)
        self.assertEqual([p['floor_count'] for p in saved['levels']],[32,96,32])
        values=Projection(saved,8)([torch.zeros(2,100,8,8) for _ in range(3)],'fusion')
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(float(values.abs().max()),0.)

    def test_accumulation_and_resume_with_new_objects(self):
        config={'channels':12,'grid':8,'flow_steps':2,'hidden_ratio':1.,'conv3x3_only':False,
                'batch_size':4,'micro_batch_size':2}
        seed_all(9)
        model=ProjectedFlow(config)
        features=torch.randn(8,12,8,8)
        full=copy.deepcopy(model)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        full_opt=torch.optim.Adam(full.parameters(),lr=.001)
        generator=torch.Generator().manual_seed(9)
        train_epoch(model,features,opt,generator,config)
        train_epoch(full,features,full_opt,torch.Generator().manual_seed(9),{**config,'micro_batch_size':4})
        for a,b in zip(model.parameters(),full.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
        state=copy.deepcopy(model.state_dict())
        optimizer_state=copy.deepcopy(opt.state_dict())
        shuffle=generator.get_state().clone()
        expected=train_epoch(model,features,opt,generator,config)
        restored=ProjectedFlow(config)
        restored.load_state_dict(state)
        resumed_opt=torch.optim.Adam(restored.parameters(),lr=.001)
        resumed_opt.load_state_dict(optimizer_state)
        resumed_generator=torch.Generator()
        resumed_generator.set_state(shuffle)
        actual=train_epoch(restored,features,resumed_opt,resumed_generator,config)
        self.assertEqual(expected,actual)
        for a,b in zip(model.state_dict().values(),restored.state_dict().values()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
