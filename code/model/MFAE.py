import torch
import torch.nn as nn
import torch.nn.functional as F
from .trm import *
import pandas as pd
import json
from .attention import *
from mamba_ssm import Mamba
Mamba = Mamba(
                    d_model=128, 
                    d_state=16,  
                    d_conv=4,    
                    expand=2,    
                )

def forward_diffusion(x, timesteps):

    noise_std = 1.0 / (timesteps ** 0.5)  
    noise = torch.normal(mean=0, std=noise_std, size=x.size(), device=x.device)
    return x + noise

class DiffusionModel(nn.Module):
    def __init__(self, feature_dim, timesteps, clip_target, hubert_target):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.feature_dim = feature_dim
        self.clip_target = clip_target 
        self.hubert_target = hubert_target  

        # 去噪网络
        self.denoise_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def reverse_diffusion(self, noisy_fused_feature, text_feature, vision_feature, audio_feature):
        x = noisy_fused_feature
        for t in range(self.timesteps, 0, -1):
            similarity_clip = F.cosine_similarity(vision_feature, self.clip_target, dim=-1)
            similarity_audio = F.cosine_similarity(audio_feature, self.hubert_target, dim=-1)

            vision_feature_adjusted = vision_feature + (1 - similarity_clip.view(-1, 1)) * (self.clip_target - vision_feature)
            audio_feature_adjusted = audio_feature + (1 - similarity_audio.view(-1, 1)) * (self.hubert_target - audio_feature)

            fused_feature = (vision_feature_adjusted + audio_feature_adjusted + text_feature) / 3

            x = self.denoise_net(fused_feature)
            x = x - torch.randn_like(x) / t  

        return x

class AE(torch.nn.Module):
    def __init__(self,dataset):
        super(AE,self).__init__()
        if dataset=='fakett':
            self.encoded_text_semantic_fea_dim=512 
        elif dataset=='fakesv':
            self.encoded_text_semantic_fea_dim=768 
        self.input_visual_frames=83

        self.latent_dim = 128
        self.steps = 100

        
        self.mlp_text_emo = nn.Sequential(nn.Linear(768,128),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,128),nn.ReLU(),nn.Dropout(0.1)) 
        
        self.mlp_img = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.1))
    
        self.mlp_audio = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.ReLU(),nn.Dropout(0.1))
        
        self.co_attention_tv = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,
                                        visual_len=self.input_visual_frames, sen_len=512, fea_v=128, fea_s=128, pos=False)
        
        self.trm_emo=nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)
        self.trm_semantic=nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)

        self.content_classifier = nn.Sequential(nn.Linear(128*3,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,2))
        

    def forward(self,**kwargs):
        all_phrase_semantic_fea=kwargs['all_phrase_semantic_fea']
        all_phrase_emo_fea=kwargs['all_phrase_emo_fea']
        raw_visual_frames=kwargs['raw_visual_frames']
        raw_audio_emo=kwargs['raw_audio_emo']


        raw_t_fea_emo=self.mlp_text_emo(all_phrase_emo_fea).unsqueeze(1)
        raw_a_fea_emo=self.mlp_audio(raw_audio_emo).unsqueeze(1) 
        fusion_emo_fea=self.trm_emo(torch.cat((raw_t_fea_emo,raw_a_fea_emo),1))

        fusion_emo_fea=torch.mean(fusion_emo_fea,1)

        raw_t_fea_semantic=self.mlp_text_semantic(all_phrase_semantic_fea)
        raw_v_fea=self.mlp_img(raw_visual_frames)

        text_fea = raw_t_fea_emo + raw_t_fea_semantic

        
        text_pooled = torch.mean(text_fea,1)
        vision_pooled = torch.mean(raw_v_fea,1)
        audio_pooled = torch.mean(raw_a_fea_emo,1)
        diff_fuse = (text_pooled + vision_pooled + audio_pooled) / 3  # 形状 (128, 128)

        content_v, content_t = self.co_attention_tv(v=raw_v_fea, s=raw_t_fea_semantic, v_len=raw_v_fea.shape[1],s_len=raw_t_fea_semantic.shape[1])

        
        diffusion_model = DiffusionModel(self.latent_dim, timesteps=self.steps, clip_target=torch.mean(content_v,1), hubert_target=fusion_emo_fea)
        diffusion_model = diffusion_model.to('cuda')
        noisy_fused_fea = forward_diffusion(diff_fuse, self.steps)

        fused_diff = diffusion_model.reverse_diffusion(noisy_fused_fea, text_pooled, vision_pooled, audio_pooled)

        content_v=torch.mean(content_v,-2)
        content_t=torch.mean(content_t,-2)
        fusion_semantic_fea=self.trm_semantic(torch.cat((content_t.unsqueeze(1),content_v.unsqueeze(1)),1))
        fusion_semantic_fea=torch.mean(fusion_semantic_fea,1) 

        AE_fea=torch.cat((fusion_emo_fea,fusion_semantic_fea, fused_diff),1)

        output_AE=self.content_classifier(AE_fea)
        return output_AE

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PosEncoding_fix(nn.Module):
    def __init__(self,  d_word_vec):
        super(PosEncoding_fix, self).__init__()
        self.d_word_vec=d_word_vec
        self.w_k=np.array([1/(np.power(10000,2*(i//2)/d_word_vec)) for i in range(d_word_vec)])

    def forward(self, inputs):
        
        pos_embs=[]
        for pos in inputs:
            pos_emb=torch.tensor([self.w_k[i]*pos.cpu() for i in range(self.d_word_vec)])
            if pos !=0:
                pos_emb[0::2]=np.sin(pos_emb[0::2])
                pos_emb[1::2]=np.cos(pos_emb[1::2])
                pos_embs.append(pos_emb)
            else:
                pos_embs.append(torch.zeros(self.d_word_vec))
        pos_embs=torch.stack(pos_embs)
        return pos_embs.cuda()

class DurationEncoding(nn.Module):
    def __init__(self,dim,dataset):
        super(DurationEncoding,self).__init__()
        if dataset=='fakett':
            with open('', 'r') as json_file:
                seg_dura_info=json.load(json_file)
        elif dataset=='fakesv':

            with open('n', 'r') as json_file:
                seg_dura_info=json.load(json_file)
        
        self.all_seg_duration=seg_dura_info['all_seg_duration']
        self.all_seg_dura_ratio=seg_dura_info['all_seg_dura_ratio']
        self.absolute_bin_edges=torch.quantile(torch.tensor(self.all_seg_duration).to(torch.float64),torch.range(0,1,0.01).to(torch.float64)).cuda()
        self.relative_bin_edges=torch.quantile(torch.tensor( self.all_seg_dura_ratio).to(torch.float64),torch.range(0,1,0.02).to(torch.float64)).cuda()
        self.ab_duration_embed=torch.nn.Embedding(101,dim)
        self.re_duration_embed=torch.nn.Embedding(51,dim)

        

        self.ocr_all_seg_duration=seg_dura_info['ocr_all_seg_duration']
        self.ocr_all_seg_dura_ratio=seg_dura_info['ocr_all_seg_dura_ratio']
        self.ocr_absolute_bin_edges=torch.quantile(torch.tensor(self.ocr_all_seg_duration).to(torch.float64),torch.range(0,1,0.01).to(torch.float64)).cuda()
        self.ocr_relative_bin_edges=torch.quantile(torch.tensor( self.ocr_all_seg_dura_ratio).to(torch.float64),torch.range(0,1,0.02).to(torch.float64)).cuda()
        self.ocr_ab_duration_embed=torch.nn.Embedding(101,dim)
        self.ocr_re_duration_embed=torch.nn.Embedding(51,dim)

        self.result_dim=dim

    def forward(self,time_value,attribute):
        all_segs_embedding=[]
        if attribute=='natural_ab':
            for dv in time_value:
                bucket_indice=torch.searchsorted(self.absolute_bin_edges, torch.tensor(dv,dtype=torch.float64))
                dura_embedding=self.ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute=='natural_re':
            for dv in time_value:
                bucket_indice=torch.searchsorted(self.relative_bin_edges, torch.tensor(dv,dtype=torch.float64))
                dura_embedding=self.re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute=='ocr_ab':
            for dv in time_value:
                bucket_indice=torch.searchsorted(self.ocr_absolute_bin_edges, torch.tensor(dv,dtype=torch.float64))
                dura_embedding=self.ocr_ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
                
        elif attribute=='ocr_re':
            for dv in time_value:
                bucket_indice=torch.searchsorted(self.ocr_relative_bin_edges, torch.tensor(dv,dtype=torch.float64))
                dura_embedding=self.ocr_re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
                

        if len(all_segs_embedding)==0:
            return torch.zeros((1,self.result_dim)).cuda() 
        return torch.stack(all_segs_embedding,dim=0).cuda() 


def get_dura_info_visual(segs,fps,total_frame):
    duration_frames=[]
    duration_time=[]
    for seg in segs:
        if seg[0]==-1 and seg[1]==-1:
            continue
        if seg[0]==0 and seg[1]==0:
            continue
        else:
            duration_frames.append(seg[1]-seg[0]+1)
            duration_time.append((seg[1]-seg[0]+1)/fps)
    duration_ratio=[min(dura/total_frame,1) for dura in duration_frames]
    return torch.tensor(duration_time).cuda(),torch.tensor(duration_ratio).cuda()


class TE(torch.nn.Module):
    def __init__(self,dataset):
        super(TE,self).__init__()
        self.input_visual_frames=83
        self.pad_seg_count=83
        self.pad_ocr_phrase_count=80

        self.ocr_pattern_fea_downscaling = nn.Sequential(
            nn.Conv2d(256, 256 //4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(256 // 4),
            nn.GELU(),
            nn.Conv2d(256 // 4, 256 // 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.mlp_ocr_pattern=nn.Sequential(nn.Linear(4096,2048),nn.ReLU(),nn.Dropout(0.1),torch.nn.Linear(2048, 512),torch.nn.ReLU(),nn.Dropout(0.1),torch.nn.Linear(512, 128),torch.nn.ReLU())
        self.proj_t=nn.Linear(512,128) 
        self.t_interseg_attention = Attention(128,heads=4,dim_head=64)

        self.proj_v=nn.Linear(512,128)
        self.intraseg_att_v=Attention(dim=128,heads=4)
        self.v_interseg_attention = Attention(dim=128,heads=4)


        self.position_encoder=PosEncoding_fix(128)
        self.dura_encoder=DurationEncoding(64,dataset)

        self.narative_interact_trm=nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)

        self.narative_classifier = nn.Sequential(nn.Linear(128*2,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,2))

    def segment_feature_aggregation_att(self,frame_fea, frames_seg_indicator):
        seg_counts = torch.bincount(frames_seg_indicator[frames_seg_indicator != -1])
        max_frames = torch.max(seg_counts).item()
        unique_segments = torch.unique(frames_seg_indicator[frames_seg_indicator != -1])

        padded_seg_frames=[]
        for idx, seg_id in enumerate(unique_segments):
            frames = frame_fea[frames_seg_indicator == seg_id]
            pad_amount = max_frames - frames.size(0)
            if pad_amount > 0:
                frames = F.pad(frames, (0, 0, 0, pad_amount), "constant", 0)
            padded_seg_frames.append(frames)

        padded_seg_frames = torch.stack(padded_seg_frames).cuda()
        aggregated_seg_fea=self.intraseg_att_v(padded_seg_frames)
        return  torch.mean(aggregated_seg_fea,1)

    def forward(self,**kwargs):
        ocr_phrases_fea=kwargs['ocr_phrase_fea']
        ocr_time_region=kwargs['ocr_time_region']
        visual_frames_fea=kwargs['visual_frames_fea']
        visual_frames_seg_indicator=kwargs['visual_frames_seg_indicator']
        visual_seg_paded=kwargs['visual_seg_paded']
        fps=kwargs['fps']
        total_frames=kwargs['total_frame']
        ocr_pattern_fea=kwargs['ocr_pattern_fea']
        

        down_scaling_ocr_pattern_fea=self.ocr_pattern_fea_downscaling(ocr_pattern_fea)
        flatten_ocr_pattern_fea=down_scaling_ocr_pattern_fea.view(down_scaling_ocr_pattern_fea.size(0), -1)
        ocr_layout_pattern=self.mlp_ocr_pattern(flatten_ocr_pattern_fea) 
        
        v_temporal=[]
        narrative_v_fea=self.proj_v(visual_frames_fea)
        for v_idx in range(len(narrative_v_fea)):
            v_seg_fea=self.segment_feature_aggregation_att(narrative_v_fea[v_idx],visual_frames_seg_indicator[v_idx]) 
            v_ab_value,v_re_value=get_dura_info_visual(visual_seg_paded[v_idx],fps[v_idx],total_frames[v_idx])
            v_ab_emb=self.dura_encoder(v_ab_value,'natural_ab') 
            v_re_emb=self.dura_encoder(v_re_value,'natural_re') 
            dura_emd=torch.cat([v_ab_emb,v_re_emb],dim=1) 
            seg_general_fea=v_seg_fea+dura_emd 
            
            #add position embedding
            seg_index=torch.tensor([i for i in range(v_seg_fea.shape[0])]).cuda()
            seg_position_embedding=self.position_encoder(seg_index) 
            seg_general_fea=v_seg_fea+seg_position_embedding
            if seg_general_fea.shape[0]<self.pad_seg_count:
                pad_seg=torch.zeros((self.pad_seg_count-seg_general_fea.shape[0],128)).cuda()
                seg_general_fea=torch.cat([seg_general_fea,pad_seg],dim=0)
            v_temporal.append(seg_general_fea)
        v_temporal=torch.stack(v_temporal,dim=0)
        device = v_temporal.device
        Mamba.to(device)
        v_temporal_mamba = Mamba(v_temporal)
        v_temporal = v_temporal + v_temporal_mamba

        t_temporal=[]
        for v_idx in range(len(ocr_phrases_fea)):
            ocr_phrase_fea=self.proj_t(ocr_phrases_fea[v_idx]) 
            ocr_ab_value,ocr_re_value=get_dura_info_visual(ocr_time_region[v_idx],fps[v_idx],total_frames[v_idx])
            ocr_ab_emb=self.dura_encoder(ocr_ab_value,'ocr_ab')
            ocr_re_emb=self.dura_encoder(ocr_re_value,'ocr_re') 
            ocr_dura_emb=torch.cat([ocr_ab_emb,ocr_re_emb],dim=1) 
            ocr_phrase_fea=ocr_phrase_fea[:ocr_dura_emb.shape[0]]
            ocr_word_fea=ocr_phrase_fea+ocr_dura_emb
            #add position embedding
            phrase_index=torch.tensor([i for i in range(ocr_re_emb.shape[0])]).cuda()
            phrase_position_embedding=self.position_encoder(phrase_index)
            ocr_word_fea=ocr_word_fea+phrase_position_embedding
        
            if ocr_word_fea.shape[0]<self.pad_ocr_phrase_count:
                pad_phrase=torch.zeros((self.pad_ocr_phrase_count-ocr_word_fea.shape[0],128)).cuda()
                ocr_word_fea=torch.cat((ocr_word_fea,pad_phrase),dim=0)
            t_temporal.append(ocr_word_fea)
        t_temporal=torch.stack(t_temporal,dim=0)
        t_temporal_mamba = Mamba(t_temporal)
        t_temporal = t_temporal + t_temporal_mamba 


        narative_t=self.t_interseg_attention(t_temporal)
        ocr_seg_count=torch.tensor([len(ocr_time_region[i]) for i in range(len(ocr_time_region))]).cuda()
        narative_t=torch.sum(narative_t,dim=1)/ocr_seg_count.unsqueeze(1)

        
        narrative_v=self.v_interseg_attention(v_temporal)
        v_seg_count=torch.tensor([len(visual_seg_paded[i]) for i in range(len(visual_seg_paded))]).cuda()
        narrative_v=torch.sum(narrative_v,dim=1)/v_seg_count.unsqueeze(1) 


        narrative_multimodal_segs_fea=torch.cat((narative_t.unsqueeze(1),narrative_v.unsqueeze(1)),1)
        narrative_multimodal_segs_fea=self.narative_interact_trm(narrative_multimodal_segs_fea)
        narrative_temporal_fea=torch.mean(narrative_multimodal_segs_fea,dim=1) 

        TE_fea=torch.cat((ocr_layout_pattern,narrative_temporal_fea),1) 
        output_TE=self.narative_classifier(TE_fea)
        return output_TE

class SimpleSSM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # 简单前馈网络代替复杂SSM，避免维度问题
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class FinalFusion(torch.nn.Module):
    def __init__(self, dataset):
        super(FinalFusion, self).__init__()
        self.content_branch = AE(dataset=dataset)  # Efusion
        self.editing_branch = TE(dataset=dataset)  # Etem
        

        self.linear_efusion_1 = nn.Linear(2, 16)
        self.linear_efusion_2 = nn.Linear(2, 16)
        self.linear_etem_1 = nn.Linear(2, 16)
        self.linear_etem_2 = nn.Linear(2, 16)
        

        self.conv1d_left = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv1d_right = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        

        self.ssm = SimpleSSM(64, 64)
        

        self.sigma = nn.Sigmoid()
        

        self.final_linear = nn.Linear(64, 2)

    def forward(self, **kwargs):

        efusion_output = self.content_branch(**kwargs)  
        etem_output = self.editing_branch(**kwargs)    
        

        efusion_linear_left = self.linear_efusion_1(efusion_output)    
        efusion_linear_right = self.linear_efusion_2(efusion_output)  
        etem_linear_left = self.linear_etem_1(etem_output)            
        etem_linear_right = self.linear_etem_2(etem_output)          
        

        efusion_conv_input = efusion_linear_right.t().unsqueeze(0)    
        etem_conv_input = etem_linear_left.t().unsqueeze(0)           
        

        conv_left = self.conv1d_left(efusion_conv_input)              
        conv_right = self.conv1d_right(etem_conv_input)              
        

        conv_left_features = conv_left.squeeze(0).t()                 
        conv_right_features = conv_right.squeeze(0).t()               
        

        combined_features = torch.cat([conv_left_features, conv_right_features], dim=1) 
        

        ssm_output = self.ssm(combined_features)                      
        

        ssm_efusion, ssm_etem = torch.chunk(ssm_output, 2, dim=1)     
        

        sigma_left_expanded = torch.cat([efusion_linear_left, torch.zeros_like(efusion_linear_left)], dim=1)  # [128, 32]
        sigma_right_expanded = torch.cat([etem_linear_right, torch.zeros_like(etem_linear_right)], dim=1)    # [128, 32]
        

        sigma_left = self.sigma(sigma_left_expanded)                  
        sigma_right = self.sigma(sigma_right_expanded)              
        

        mult_left = sigma_left * ssm_efusion                        
        mult_right = sigma_right * ssm_etem                        
        

        combined_output = torch.cat([mult_left, mult_right], dim=1)  
        

        final_output = self.final_linear(combined_output)            
        
        return final_output, efusion_output, etem_output