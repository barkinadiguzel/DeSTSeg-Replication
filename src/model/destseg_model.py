import torch
import torch.nn as nn
from ..backbone.teacher_resnet import TeacherResNet
from ..modules.student_model import StudentNet
from ..modules.segmentation_net import SegNet
from ..modules.feature_pipeline import compute_multilevel_similarity

class DeSTSegModel(nn.Module):
    def __init__(self, pretrained_teacher=True, freeze_teacher=True, freeze_student_seg=True):
        super().__init__()
        self.teacher = TeacherResNet(pretrained=pretrained_teacher)
        self.student = StudentNet()
        self.segnet = SegNet()
        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad=False
        if freeze_student_seg: # for step 2 ofc
            for p in self.student.parameters():
                p.requires_grad=False
            for p in self.segnet.parameters():
                p.requires_grad=True 

    def forward(self, x_clean, x_anom):
        T_feats = self.teacher(x_clean)
        S_feats = self.student(x_anom)

        X = compute_multilevel_similarity(T_feats, S_feats)
        seg_out = self.segnet(X)
        return seg_out, T_feats, S_feats
