import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics
from self_attention import EncoderLayer
from dataset import PT_FEATURE_SIZE
import operator

SMILESCLen = 64    # SMILES Char Num


class Squeeze(nn.Module):   #Dimention Module
    def forward(self, input: torch.Tensor):
        return input.squeeze()

class DilatedConv(nn.Module):     # Dilated Convolution
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DilatedConvBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)  # Down Dimention
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)    # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)     # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)     # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)     # Dilated scale:8(2^3)
        self.d16 = DilatedConv(n, n, 3, 1, 16)   # Dilated scale:16(2^4)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)

        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class DilatedConvBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)  # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)   # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)   # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)   # Dilated scale:8(2^3)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):

        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class CAPLA(nn.Module):
    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128

        seq_oc = 128
        pkt_oc = 64
        smi_oc = 128
        td_oc = 32

        # SMILES, POCKET, PROTEIN Embedding
        self.smi_embed = nn.Embedding(SMILESCLen, smi_embed_size)
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)


        # Global DilatedConv Module
        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedConvBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        # Pocket DilatedConv Module
        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3))
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)

        
        td_conv = []
        ic = 1
        for oc in [16, 32, td_oc * 2]:
            td_conv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        td_conv.append(nn.AdaptiveMaxPool1d(1))
        td_conv.append(Squeeze())
        self.td_conv = nn.Sequential(*td_conv)


        td_onlyconv = []
        ic = 1
        for oc in [16, 32, td_oc]:
            td_onlyconv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        self.td_onlyconv = nn.Sequential(*td_onlyconv)


        # Ligand DilatedConv Module
        conv_smi = []
        ic = smi_embed_size

        
        # Cross-Attention Module
        self.smi_attention_poc = EncoderLayer(128, 128, 0.1, 0.1, 2)  # 注意力机制
        self.tdpoc_attention_tdlig = EncoderLayer(32, 64, 0.1, 0.1, 1)
        
        self.adaptmaxpool = nn.AdaptiveMaxPool1d(1)
        self.squeeze = Squeeze()

        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedConvBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)


        # Dropout
        self.cat_dropout = nn.Dropout(0.2)
        # FNN
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc + pkt_oc + smi_oc, 256),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, seq, pkt, smi):
        # D(B_s,N,L)
        seq_embed = self.seq_embed(seq)
        seq_embed = torch.transpose(seq_embed, 1, 2)
        seq_conv = self.conv_seq(seq_embed)

        pkt_embed = self.seq_embed(pkt)
        smi_embed = self.smi_embed(smi)
        smi_attention = smi_embed

        
        sminp = smi_embed.cpu().numpy()
        pktnp = pkt_embed.cpu().numpy()
        np.save("v2013smiembed.npy",sminp)
        np.save("v2013pktembed.npy",pktnp)

        smi_embed = self.smi_attention_poc(smi_embed, pkt_embed)
        pkt_embed = self.smi_attention_poc(pkt_embed, smi_attention)


#        print(smi_embed.shape)
#        print(pkt_embed.shape)
#        sminp = smi_embed.cpu().numpy()
#        pktnp = pkt_embed.cpu().numpy()
#        np.save("V2013smiAtten.npy",sminp)
#        np.save("V2013pktAtten.npy",pktnp)

        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)


        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)

        concat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)
        #Attention Analyse change dim from 1 to 0

        concat = self.cat_dropout(concat)

        output = self.classifier(concat)
        return output

# Model Test
def test(model: nn.Module, test_loader, loss_function, device, show):
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        model.eval()
        for idx, (*x,y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
        
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            
            temp = x[1].cpu().numpy()
            np.save("data.npy",temp)
            y_hat = model(*x)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    # Save Result OR Analyse
#    np.save("set1_outputs.npy", outputs)
#    np.save("targets.npy", targets)

    # loss
    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation

