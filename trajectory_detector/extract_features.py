import numpy as np
import pandas as pd
import collections
import os
import functools
import json
import sys
import logging
import pickle
import re
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

# Import local modules
from models import FusionModel
from dataset import sampleDataset, collate_fn

def extract_features(data_dir, ckpt_path, output_file, batch_size=32):
    device = torch.device("cpu") # Force CPU for macOS stability
    print(f"Using device: {device}")

    # Configs from angle_pretrain.py
    use_merge_split = True
    suffix = "_merge_split"
    
    # Load vocab and embeddings to get sizes
    vocab_name = "w2v/vocab%s.pk"%(suffix)
    print(f"Loading vocab from {os.path.join(data_dir, vocab_name)}")
    mouse_vocab, mouse_token2idx, location_vocab, location_token2idx = pickle.load(open(os.path.join(data_dir, vocab_name), "rb"))
    
    # We need the embedding sizes, but we don't necessarily need the weights 
    # since we load them from the checkpoint anyway.
    # However, let's match the config.
    num_embeddings_loc = len(location_token2idx) + 1
    num_embeddings_mo = len(mouse_token2idx) + 1

    location_config = {'input_size':100,'input_size_fre':9,'hidden_size':100,'dropout_rate':0.2,'bidirectional':True,\
                       'use_rnn':True,'use_self_attention':False,'num_hidden_layers':1,'num_attention_heads':4,'use_fre':False,'use_cnn':True,\
                       'use_embedding':True,'embedding_weight':None,'num_embeddings':num_embeddings_loc,'num_cnn_layers':3,\
                       'use_time_position':False,'pre_cnn_time_position':False,'use_idx_embedding':False,'idx_embedding_weight':0,\
                       'use_geo_position':False,'mse_loss':False,'logloss':True,'sinusoidal':False}
    
    mouse_config = {'input_size':100,'input_size_fre':9,'hidden_size':100,'dropout_rate':0.2,'bidirectional':True,\
                       'use_rnn':True,'use_self_attention':False,'num_hidden_layers':1,'num_attention_heads':4,'use_fre':False,'use_cnn':True,\
                       'use_embedding':True,'embedding_weight':None,'num_embeddings':num_embeddings_mo,'num_cnn_layers':3,\
                       'use_time_position':False,'pre_cnn_time_position':False,'use_idx_embedding':False,'idx_embedding_weight':0,\
                       'use_geo_position':False,'mse_loss':False,'logloss':True,'sinusoidal':False}

    print("Initializing FusionModel...")
    model = FusionModel(location_config, mouse_config, 400, 400, 
                        use_mutual_attention=True, use_residual=True, use_rnn_output=True,
                        pretrain=False, use_embed=False, model_type='ConvGRU', tri_loss=False)

    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Loading dataset metadata...")
    file_name = "masked_action_sequence_with_feature.pickle"
    tmp = pickle.load(open(os.path.join(data_dir, file_name + suffix), "rb"))
    if len(tmp) == 3: # Handle case where it's (idx2user, user2idx, day2action)
        day2action = tmp[2]
    else:
        day2action = tmp

    # Extract user IDs in the same order as sampleDataset
    flat_ids = [e[0] for day in day2action for e in day]

    dataset = sampleDataset(day2action)
    func = functools.partial(collate_fn, use_token=True, fil=False)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=func, shuffle=False)

    all_features = []
    all_meta = []

    print(f"Extracting features for {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            X_loc = batch['location_feature'].to(device)
            X_fre_loc = batch['location_fre_feature'].to(device)
            valid_len_loc = batch['location_length'].to(device)
            time_dis_loc = batch['location_dis'].to(device)
            
            X_mo = batch['mouse_feature'].to(device)
            X_fre_mo = batch['mouse_fre_feature'].to(device)
            valid_len_mo = batch['mouse_length'].to(device)
            time_dis_mo = batch['mouse_dis'].to(device)
            
            labels = batch['labels'].to(device)
            geo_pos_loc = batch.get('location_xy').to(device) if 'location_xy' in batch else None
            
            outputs = model(X_loc, X_fre_loc, valid_len_loc, time_dis_loc,
                            X_mo, X_fre_mo, valid_len_mo, time_dis_mo, labels,
                            geo_position_ids_loc=geo_pos_loc)
            
            hidden = outputs[4]
            features_np = hidden.cpu().numpy()
            
            for i in range(len(features_np)):
                all_features.append(features_np[i])
                sample_idx = batch_idx * batch_size + i
                all_meta.append({
                    "user_id": flat_ids[sample_idx] if sample_idx < len(flat_ids) else "Unknown",
                    "label": int(labels[i].cpu().item())
                })

    # Save the output
    features_matrix = np.array(all_features)
    np.save(f"{output_file}_features.npy", features_matrix)

    with open(f"{output_file}_meta.json", "w") as f:
        json.dump(all_meta, f)

    print(f"Saved extracted features: {features_matrix.shape} to {output_file}_features.npy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D","--data_dir",type=str,help="数据文件夹",default = '/Users/williammuji/Codes/AntiCheat/T-Detector/train_data_sampled_processed/')
    parser.add_argument("-C","--ckpt",type=str,help="Checkpoint path", required=True)
    parser.add_argument("-O","--output",type=str,help="输出前缀",default = 'extracted')
    args = parser.parse_args()

    output_path = os.path.join(args.data_dir, args.output)
    extract_features(args.data_dir, args.ckpt, output_path)
