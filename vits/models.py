
import torch

from torch import nn
from torch.nn import functional as F
from vits import attentions
from vits import commons
from vits import modules
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator, Generator_old
from vits.modules_grl import SpeakerClassifier
from style_encoder.reference_encoder import ReferenceEncoder

class TextEncoder_old(nn.Module):
    def __init__(self,
                 in_channels,
                 vec_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 mel_channels,
                 num_styles):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.hub = nn.Conv1d(vec_channels, hidden_channels, kernel_size=5, padding=2)

        ### Adding style encoder
        self.re = ReferenceEncoder(mel_channels, hidden_channels)
        ### end style encoder
        self.pit = nn.Embedding(256, hidden_channels)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)


        if(num_styles > 0):
            self.use_style_guided = True
            self.classifier_layer = nn.Linear(hidden_channels, num_styles)
        else:
            self.use_style_guided = False

    def forward(self, x, x_lengths, v, f0, melspe, stl_rep = None):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        v = torch.transpose(v, 1, -1)  # [b, h, t]
        v = self.hub(v) * x_mask
        if(stl_rep is None):
            st = self.re(melspe)
        else:
            st = stl_rep

        if(self.use_style_guided):
            stl_preds = self.classifier_layer(st)
        else:
            stl_preds = None

        ##PRINT DEBUG
        # print(st.shape, v.shape, x.shape, self.pit(f0).transpose(1, 2).shape)
        x = x + v + self.pit(f0).transpose(1, 2) + st.unsqueeze(-1)
        # print(x.shape)
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask, x, stl_preds

class TextEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 vec_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.hub = nn.Conv1d(vec_channels, hidden_channels, kernel_size=5, padding=2)


        
        self.pit = nn.Embedding(256, hidden_channels)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)


    def forward(self, x, x_lengths, v, f0):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        v = torch.transpose(v, 1, -1)  # [b, h, t]
        v = self.hub(v) * x_mask
#         if(stl_rep is None):
#             st = self.re(melspe)
#         else:
#             st = stl_rep

#         if(self.use_style_guided):
#             stl_preds = self.classifier_layer(st)
#         else:
#             stl_preds = None

        ##PRINT DEBUG
        # print(st.shape, v.shape, x.shape, self.pit(f0).transpose(1, 2).shape)
        x = x + v + self.pit(f0).transpose(1, 2) 
        # print(x.shape)
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask, x

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, s =None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g,s = s, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, s=s,reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

class ResidualCouplingBlock_old(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer_old(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, s=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g, s= s)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.emb_g = nn.Linear(hp.vits.spk_dim, hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
            hp.vits.vec_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1
        )
        
        self.re = ReferenceEncoder(hp.data.mel_channels, hp.vits.style_emb_dim)
        self.emb_s = nn.Linear(hp.vits.style_emb_dim, hp.vits.gin_channels)

        self.speaker_classifier = SpeakerClassifier(
            hp.vits.hidden_channels,
            hp.vits.spk_dim,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)

    def forward(self, ppg, vec, pit, spec, spk, ppg_l, spec_l, mel_perturbed):
        ppg = ppg + torch.randn_like(ppg) * 1  # Perturbation
        vec = vec + torch.randn_like(vec) * 2  # Perturbation
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
#         print(mel_perturbed.shape)
        s = self.emb_s(F.normalize(self.re(mel_perturbed.permute(0,2,1)))).unsqueeze(-1)
#         print(s.shape)
        
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g, s=s)

        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_q, pit, spec_l, self.segment_size)
        
        audio = self.dec(spk, z_slice, pit_slice, s)

        # SNAC to flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=spk, s=s)
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk, s=s, reverse=True)
        # speaker
        spk_preds = self.speaker_classifier(x)
        stl_preds = None
        return audio, ids_slice, spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds, stl_preds

    def infer(self, ppg, vec, pit, spk, ppg_l, mel_rep, stl_rep=None):
        s = self.emb_s(F.normalize(self.re(mel_rep))).unsqueeze(-1)
        ppg = ppg + torch.randn_like(ppg) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, s = s, reverse=True)
        o = self.dec(spk, z * ppg_mask, f0=pit, s = s)
        return o
    
    
class SynthesizerInfer(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
            hp.vits.vec_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)
        self.re = ReferenceEncoder(hp.data.mel_channels, hp.vits.style_emb_dim)
        self.emb_s = nn.Linear(hp.vits.style_emb_dim, hp.vits.gin_channels)
        self.emb_g = nn.Linear(hp.vits.spk_dim, hp.vits.gin_channels)
    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.dec.remove_weight_norm()

    def pitch2source(self, f0):
        return self.dec.pitch2source(f0)

    def source2wav(self, source):
        return self.dec.source2wav(source)

    def inference(self, ppg, vec, pit,spk, ppg_l, source, mel_rep, stl_rep=None):
        if(stl_rep is not None):
            s = self.emb_s(F.normalize(stl_rep)).unsqueeze(-1)
        else:
            s = self.emb_s(F.normalize(self.re(mel_rep))).unsqueeze(-1)
            
        g = self.emb_g(F.normalize(spk))
#         print(ppg_l)
#         print(s.shape)
#         print(g.shape)
#         print(ppg.shape)
#         ppg = ppg + torch.randn_like(ppg) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, s = s, reverse=True)
#         o = self.dec(spk, z * ppg_mask, f0=pit, s = s)
#         print(z_p.shape)
#         print(z.shape)
#         print(source.shape)
        o = self.dec.inference(spk, z * ppg_mask, source, s)
#         print(o.shape)
        return o
    
    def voice_conversion(self, spec, spec_len, pit, g_src, g_tgt, s_src, s_tgt):
        g_src = g_src
        g = self.emb_g(F.normalize(g_src)).unsqueeze(-1)
        g_tgt = g_tgt
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_len, g=g, s=s_src)
#         z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src if not self.zero_g else torch.zeros_like(g_src), tau=tau)
        z_p,_ = self.flow(z_q, spec_mask, g=g_src, s=s_src)
        z_hat,_ = self.flow(z_p, spec_mask, g=g_tgt, s=s_tgt, reverse=True)
        o_hat = self.dec.inference(g_tgt, z_hat * spec_mask, pit, s_tgt)
        return o_hat

    
class SynthesizerInfer_old(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.enc_p = TextEncoder_old(
            hp.vits.ppg_dim,
            hp.vits.vec_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
            hp.data.mel_channels,
            hp.vits.num_styles
        )
        self.flow = ResidualCouplingBlock_old(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator_old(hp=hp)

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.dec.remove_weight_norm()

    def pitch2source(self, f0):
        return self.dec.pitch2source(f0)

    def source2wav(self, source):
        return self.dec.source2wav(source)

    def inference(self, ppg, vec, pit, spk, ppg_l, source, mel_rep, stl_rep = None):
        z_p, m_p, logs_p, ppg_mask, x, stl_preds = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit), melspe= mel_rep, stl_rep = stl_rep)
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o
