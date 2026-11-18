"""Teacher64 preserves a strong prior without erasing learned calendar information."""
import io,unittest
import numpy as np
import torch
from exp.exp_chronogeom_downstream import CalendarTimeBase
from exp.exp_chronogeom_calendar_level_direct63 import DirectStudent
from exp.exp_chronogeom_strong_teacher64 import PredictionTeacher

class UpdatedTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        state=CalendarTimeBase(dict(seq_len=96,pred_len=96),2024,np.zeros((168,168),np.float32)).state_dict()
        self.base=DirectStudent(2024,state,np.zeros(8),np.ones(8))
        torch.manual_seed(64001)
        with torch.no_grad():
            self.base.reader[0].weight[:,39:].normal_(std=.3)
            self.base.reader[2].weight.normal_(std=.2)
            self.base.calendar_level.weight.normal_(std=.1)
        self.x=torch.randn(3,96,1);self.cal=torch.randn(3,96,31);self.ids=torch.arange(96)[None].repeat(3,1)%72
        self.table=torch.randn(72,3)
    def test_initial_teacher_preserves_complete_strong_prior(self):
        teacher=PredictionTeacher(self.base,self.table)
        other=PredictionTeacher(self.base,-3*self.table)
        a=self.base(self.x,self.cal,self.ids)
        b=teacher(self.x,self.cal,self.ids)
        self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6))
        self.assertTrue(torch.equal(b,other(self.x,self.cal,self.ids)))
        self.assertEqual(sum(p.numel() for p in teacher.parameters()),995)
        self.assertEqual(sum(p.numel() for p in teacher.parameters() if p.requires_grad),705)
    def test_reader_can_learn_knowledge_without_updating_numeric_or_calendar_prior(self):
        teacher=PredictionTeacher(self.base,self.table)
        y=torch.randn(3,96,1)
        loss=(teacher(self.x,self.cal,self.ids)-y).square().mean();loss.backward()
        self.assertGreater(float(teacher.reader[0].weight.grad[:,39:].abs().sum()),0)
        self.assertTrue(all(p.grad is None for n,p in teacher.named_parameters() if not n.startswith("reader.")))
    def test_knowledge_changes_shape_not_mean_after_training(self):
        teacher=PredictionTeacher(self.base,self.table)
        with torch.no_grad():teacher.reader[0].weight[:,39:].normal_(std=.1)
        p=teacher(self.x,self.cal,self.ids)
        level=self.base.level_prediction(self.x,self.cal)
        self.assertTrue(torch.allclose(p.mean(1,keepdim=True),level,atol=1e-6,rtol=1e-6))
        self.assertGreater(float((p-self.base(self.x,self.cal,self.ids)).square().mean()),1e-9)
    def test_export_has_prior_and_knowledge_without_llm_or_target(self):
        teacher=PredictionTeacher(self.base,self.table)
        with torch.no_grad():teacher.reader[0].weight[:,39:].normal_(std=.1)
        stream=io.BytesIO();torch.save(teacher.state_dict(),stream);stream.seek(0)
        restored=PredictionTeacher(self.base,torch.zeros_like(self.table))
        restored.load_state_dict(torch.load(stream,weights_only=True))
        self.assertTrue(torch.equal(teacher(self.x,self.cal,self.ids),restored(self.x,self.cal,self.ids)))

if __name__=="__main__":unittest.main()
