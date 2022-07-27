import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()



        self.projection_matrix1 = torch.nn.Parameter(torch.randn(32, 5)) #7 128
        self.projection_matrix2 = torch.nn.Parameter(torch.randn(64, 5))
        self.projection_matrix3 = torch.nn.Parameter(torch.randn(32, 5))
        self.projection_matrix4 = torch.nn.Parameter(torch.randn(64, 5))
        self.projection_matrix5 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix6 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix7 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix8 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix9 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix10 = torch.nn.Parameter(torch.randn(2, 5))
        self.projection_matrix11 = torch.nn.Parameter(torch.randn(2, 5))
        self.projection_matrix12 = torch.nn.Parameter(torch.randn(2, 5))
        self.projection_matrix13 = torch.nn.Parameter(torch.randn(2, 5))
        self.projection_matrix14 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix15 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix16 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix17 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix18 = torch.nn.Parameter(torch.randn(4, 5))
        self.projection_matrix19 = torch.nn.Parameter(torch.randn(128, 5))
        self.projection_matrix20 = torch.nn.Parameter(torch.randn(4, 5))
        self.projection_matrix21 = torch.nn.Parameter(torch.randn(128, 5))
        self.projection_matrix22 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix23 = torch.nn.Parameter(torch.randn(256, 5))
        self.projection_matrix24 = torch.nn.Parameter(torch.randn(256, 5))



        self.block_1 = nn.LSTM(input_size=24*5, hidden_size=300, num_layers=2, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(333*300*2, 1, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )


    def forward(self, X):
        batch, seq, _ = X.size()
        l_1 = X[:, :, 0:32]
        l_2 = X[:, :, 32:96]
        l_3 = X[:, :, 96:128]
        l_4 = X[:, :, 128:192]
        l_5 = X[:, :, 192:448]
        l_6 = X[:, :, 448:704]
        l_7 = X[:, :, 704:960]
        l_8 = X[:, :, 960:1216]
        l_9 = X[:, :, 1216:1472]
        l_10 = X[:, :, 1472:1474]
        l_11 = X[:, :, 1474:1476]
        l_12 = X[:, :, 1476:1478]
        l_13 = X[:, :, 1478:1480]
        l_14 = X[:, :, 1480:1736]
        l_15 = X[:, :, 1736:1992]
        l_16 = X[:, :, 1992:2248]
        l_17 = X[:, :, 2248:2504]
        l_18 = X[:, :, 2504:2508]
        l_19 = X[:, :, 2508:2636]
        l_20 = X[:, :, 2636:2640]
        l_21 = X[:, :, 2640:2768]
        l_22 = X[:, :, 2768:3024]
        l_23 = X[:, :, 3024:3280]
        l_24 = X[:, :, 3280:3536]
        l_1_embedding = l_1.matmul(self.projection_matrix1)
        l_2_embedding = l_2.matmul(self.projection_matrix2)
        l_3_embedding = l_3.matmul(self.projection_matrix3)
        l_4_embedding = l_4.matmul(self.projection_matrix4)
        l_5_embedding = l_5.matmul(self.projection_matrix5)
        l_6_embedding = l_6.matmul(self.projection_matrix6)
        l_7_embedding = l_7.matmul(self.projection_matrix7)
        l_8_embedding = l_8.matmul(self.projection_matrix8)
        l_9_embedding = l_9.matmul(self.projection_matrix9)
        l_10_embedding = l_10.matmul(self.projection_matrix10)
        l_11_embedding = l_11.matmul(self.projection_matrix11)
        l_12_embedding = l_12.matmul(self.projection_matrix12)
        l_13_embedding = l_13.matmul(self.projection_matrix13)
        l_14_embedding = l_14.matmul(self.projection_matrix14)
        l_15_embedding = l_15.matmul(self.projection_matrix15)
        l_16_embedding = l_16.matmul(self.projection_matrix16)
        l_17_embedding = l_17.matmul(self.projection_matrix17)
        l_18_embedding = l_18.matmul(self.projection_matrix18)
        l_19_embedding = l_19.matmul(self.projection_matrix19)
        l_20_embedding = l_20.matmul(self.projection_matrix20)
        l_21_embedding = l_21.matmul(self.projection_matrix21)
        l_22_embedding = l_22.matmul(self.projection_matrix22)
        l_23_embedding = l_23.matmul(self.projection_matrix23)
        l_24_embedding = l_24.matmul(self.projection_matrix24)

        X_projected = torch.cat((l_1_embedding, l_2_embedding, l_3_embedding, l_4_embedding, l_5_embedding, l_6_embedding, l_7_embedding, l_8_embedding, l_9_embedding, l_10_embedding, l_11_embedding, l_12_embedding, l_13_embedding, l_14_embedding, l_15_embedding, l_16_embedding, l_17_embedding, l_18_embedding, l_19_embedding, l_20_embedding, l_21_embedding, l_22_embedding, l_23_embedding, l_24_embedding), -1)

        return X_projected

