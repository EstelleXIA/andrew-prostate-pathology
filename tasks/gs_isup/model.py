"""Ordinal Gleason/ISUP grading models (CORAL).

``GSEncoder`` (and its ``Attention_net_gated`` dependency) are imported from
``common.layers``. The architecture comprises a ``down=4`` bottleneck, dual
primary/secondary heads, ``classifiers_1`` / ``classifiers_2`` ModuleLists, a
multi-head-attention gating branch, a deterministic ISUP mapping matrix, and the
CoralLayer instance branch.
"""
import torch
import torch.nn as nn
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss

from common.layers import GSEncoder


class ProstateGradingModel(nn.Module):
    def __init__(self, num_classes_gleason=3, input_dim=1280, with_low_input=True, with_high_input=True):
        """
        num_classes_gleason: number of Gleason grades (3, 4, 5).
        """
        super().__init__()
        self.with_low = with_low_input
        self.with_high = with_high_input
        self.feat_dim = input_dim
        down = 4
        self.mapping_x_256 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // down),
                                           nn.ReLU(),
                                           nn.Dropout(0.1))
        if self.with_low:
            self.mapping_x_512 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // down),
                                               nn.ReLU(),
                                               nn.Dropout(0.1))

        self.backbone_primary = GSEncoder(patch_dim=self.feat_dim // down)  # feature-extraction backbone
        self.backbone_secondary = GSEncoder(patch_dim=self.feat_dim // down)  # feature-extraction backbone

        self.primary_head = nn.Linear(self.feat_dim // down, num_classes_gleason)
        self.secondary_head = nn.Linear(self.feat_dim // down, num_classes_gleason)

        # ISUP probability layer (no trainable parameters)
        self.logits_to_probas = nn.Softmax(dim=1)
        self.isup_matrix = self._build_isup_mapping_matrix()

        self.n_classes = num_classes_gleason

        bag_classifiers_1 = [nn.Linear(self.feat_dim // down, 1) for i in range(self.n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers_1 = nn.ModuleList(bag_classifiers_1)

        bag_classifiers_2 = [nn.Linear(self.feat_dim // down, 1) for i in range(self.n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers_2 = nn.ModuleList(bag_classifiers_2)

        if self.with_low and self.with_high:
            self.multihead_attn = nn.MultiheadAttention(embed_dim=self.feat_dim // down, num_heads=4, batch_first=True)

        self.instance_classifiers = CoralLayer(size_in=self.feat_dim // down, num_classes=self.n_classes)
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def _build_isup_mapping_matrix(self):
        """
        Build the deterministic Gleason-combination -> ISUP mapping matrix.
        ISUP rules:
          3+3 -> 1, 3+4 -> 2, 4+3 -> 3, 4+4/3+5/5+3 -> 4, 4+5/5+4/5+5 -> 5
        """
        matrix = torch.zeros(5, 3, 3)  # [ISUP, Primary, Secondary]
        matrix[0, 0, 0] = 1  # 3+3 -> ISUP1
        matrix[1, 0, 1] = 1  # 3+4 -> ISUP2
        matrix[2, 1, 0] = 1  # 4+3 -> ISUP3
        matrix[3, 1, 1] = 1  # 4+4 -> ISUP4
        matrix[3, 0, 2] = 1  # 3+5 -> ISUP4
        matrix[3, 2, 0] = 1  # 5+3 -> ISUP4
        matrix[4, 1, 2] = 1  # 4+5 -> ISUP5
        matrix[4, 2, 1] = 1  # 5+4 -> ISUP5
        matrix[4, 2, 2] = 1  # 5+5 -> ISUP5
        return matrix  # fixed mapping; no gradients

    def forward(self, x_256, x_512, instance_label_256, instance_label_512, wsi_latent):

        feature_morph_256 = self.mapping_x_256(x_256)
        primary_features_256, A_primary_256, x_input_256 = self.backbone_primary(feature_morph_256)
        secondary_features_256, _, _ = self.backbone_secondary(feature_morph_256)

        total_inst_loss = 0.0

        if self.with_low and self.with_high:
            feature_morph_512 = self.mapping_x_512(x_512)
            primary_features_512, A_primary_512, x_input_512 = self.backbone_primary(feature_morph_512)
            secondary_features_512, _, _ = self.backbone_secondary(feature_morph_512)

            primary_features, _ = self.multihead_attn(query=primary_features_256.unsqueeze(1),
                                                      key=primary_features_512.unsqueeze(1),
                                                      value=primary_features_512.unsqueeze(1))
            secondary_features, _ = self.multihead_attn(query=secondary_features_256.unsqueeze(1),
                                                        key=secondary_features_512.unsqueeze(1),
                                                        value=secondary_features_512.unsqueeze(1))
        else:
            if self.with_low:
                feature_morph_512 = self.mapping_x_512(x_512)
                primary_features_512, A_primary_512, x_input_512 = self.backbone_primary(feature_morph_512)
                secondary_features_512, _, _ = self.backbone_secondary(feature_morph_512)

                primary_features = primary_features_512
                secondary_features = secondary_features_512
            else:
                primary_features = primary_features_256
                secondary_features = secondary_features_256

        primary_logits = torch.empty(1, self.n_classes).to(primary_features.dtype).to(primary_features.device)
        for c in range(3):
            primary_logits[0, c] = self.classifiers_1[c](primary_features[c])

        secondary_logits = torch.empty(1, self.n_classes).to(secondary_features.dtype).to(secondary_features.device)
        for c in range(3):
            secondary_logits[0, c] = self.classifiers_2[c](secondary_features[c])

        primary_proba = self.logits_to_probas(primary_logits)
        secondary_proba = self.logits_to_probas(secondary_logits)

        # joint probability: P(Pri, Sec) = P(Pri) * P(Sec)
        joint_proba = torch.einsum('bi,bj->bij', primary_proba, secondary_proba)  # [B, 3, 3]

        # map to ISUP probability: P(ISUP=k) = sum_{Pri,Sec} P(Pri,Sec) * I(Pri,Sec->k)
        isup_proba = torch.einsum('bij,kij->bk', joint_proba, self.isup_matrix.to(x_input_256.device))

        primary_proba_ordinal = torch.flip(torch.flip(primary_proba, dims=[1]).cumsum(dim=1), dims=[1])[:, 1:]
        secondary_proba_ordinal = torch.flip(torch.flip(secondary_proba, dims=[1]).cumsum(dim=1), dims=[1])[:, 1:]
        isup_proba_ordinal = torch.flip(torch.flip(isup_proba, dims=[1]).cumsum(dim=1), dims=[1])[:, 1:]

        return primary_proba_ordinal, secondary_proba_ordinal, isup_proba_ordinal, isup_proba, total_inst_loss


class ProstateGradingModelNuclei(nn.Module):
    def __init__(self, num_classes_gleason=3, with_prism=True, captum=False):
        """
        num_classes_gleason: number of Gleason grades (3, 4, 5).
        """
        super().__init__()
        self.captum = captum
        self.feat_dim = 1280
        self.with_prism = with_prism
        down = 4
        self.mapping_x_256 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // down),
                                           nn.ReLU(),
                                           nn.Dropout(0.1))
        self.mapping_x_512 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // down),
                                           nn.ReLU(),
                                           nn.Dropout(0.1))
        self.mapping_x_256_with_nuclei = nn.Sequential(nn.Linear(self.feat_dim // down + 85, self.feat_dim // down),
                                                       nn.ReLU(),
                                                       nn.Dropout(0.1))
        self.mapping_x_512_with_nuclei = nn.Sequential(nn.Linear(self.feat_dim // down + 85, self.feat_dim // down),
                                                       nn.ReLU(),
                                                       nn.Dropout(0.1))
        self.backbone_primary = GSEncoder(patch_dim=self.feat_dim // down)  # feature-extraction backbone
        self.backbone_secondary = GSEncoder(patch_dim=self.feat_dim // down)  # feature-extraction backbone

        self.primary_head = nn.Linear(self.feat_dim // down, num_classes_gleason)
        self.secondary_head = nn.Linear(self.feat_dim // down, num_classes_gleason)

        # ISUP probability layer (no trainable parameters)
        self.logits_to_probas = nn.Softmax(dim=1)
        self.isup_matrix = self._build_isup_mapping_matrix()

        self.n_classes = num_classes_gleason
        if self.with_prism:
            self.wsi_dim = 1280
            add_neuron = self.wsi_dim // 2
            self.transform_prism = nn.Linear(self.wsi_dim, self.wsi_dim // 2)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(p=0.2)
        else:
            add_neuron = 0

        bag_classifiers_1 = [nn.Linear(self.feat_dim // down + add_neuron, 1) for i in range(self.n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers_1 = nn.ModuleList(bag_classifiers_1)

        bag_classifiers_2 = [nn.Linear(self.feat_dim // down + add_neuron, 1) for i in range(self.n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers_2 = nn.ModuleList(bag_classifiers_2)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.feat_dim // down, num_heads=4, batch_first=True)

        self.instance_classifiers = CoralLayer(size_in=self.feat_dim // down, num_classes=self.n_classes)
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def _build_isup_mapping_matrix(self):
        """
        Build the deterministic Gleason-combination -> ISUP mapping matrix.
        ISUP rules:
          3+3 -> 1, 3+4 -> 2, 4+3 -> 3, 4+4/3+5/5+3 -> 4, 4+5/5+4/5+5 -> 5
        """
        matrix = torch.zeros(5, 3, 3)  # [ISUP, Primary, Secondary]
        matrix[0, 0, 0] = 1  # 3+3 -> ISUP1
        matrix[1, 0, 1] = 1  # 3+4 -> ISUP2
        matrix[2, 1, 0] = 1  # 4+3 -> ISUP3
        matrix[3, 1, 1] = 1  # 4+4 -> ISUP4
        matrix[3, 0, 2] = 1  # 3+5 -> ISUP4
        matrix[3, 2, 0] = 1  # 5+3 -> ISUP4
        matrix[4, 1, 2] = 1  # 4+5 -> ISUP5
        matrix[4, 2, 1] = 1  # 5+4 -> ISUP5
        matrix[4, 2, 2] = 1  # 5+5 -> ISUP5
        return matrix  # fixed mapping; no gradients

    def forward(self, x_256, x_512, instance_label_256, instance_label_512, wsi_latent):
        feature_morph_256 = self.mapping_x_256(x_256[:, :self.feat_dim])
        feature_morph_256 = torch.concat([feature_morph_256, x_256[:, self.feat_dim:]], dim=1)
        primary_features_256, A_primary_256, x_input_256 = self.backbone_primary(self.mapping_x_256_with_nuclei(feature_morph_256))
        secondary_features_256, A_secondary_256, _ = self.backbone_secondary(self.mapping_x_256_with_nuclei(feature_morph_256))

        feature_morph_512 = self.mapping_x_512(x_512[:, :self.feat_dim])
        feature_morph_512 = torch.concat([feature_morph_512, x_512[:, self.feat_dim:]], dim=1)
        primary_features_512, A_primary_512, x_input_512 = self.backbone_primary(self.mapping_x_512_with_nuclei(feature_morph_512))
        secondary_features_512, A_secondary_512, _ = self.backbone_secondary(self.mapping_x_512_with_nuclei(feature_morph_512))

        total_inst_loss = 0.0

        if len(list(instance_label_256.shape)) != 1:
            instance_logits_256 = self.instance_classifiers(x_input_256)
            levels_256 = levels_from_labelbatch(instance_label_256.squeeze(0).unsqueeze(1).type(torch.LongTensor),
                                                num_classes=self.n_classes).to(instance_logits_256.device)
            instance_logits_512 = self.instance_classifiers(x_input_512)
            levels_512 = levels_from_labelbatch(instance_label_512.squeeze(0).unsqueeze(1).type(torch.LongTensor),
                                                num_classes=self.n_classes).to(instance_logits_512.device)

            inst_loss_256 = coral_loss(instance_logits_256, levels_256)
            inst_loss_512 = coral_loss(instance_logits_512, levels_512)

            total_inst_loss = 0.5 * (inst_loss_256 + inst_loss_512)

        primary_features, _ = self.multihead_attn(query=primary_features_256.unsqueeze(1),
                                                  key=primary_features_512.unsqueeze(1),
                                                  value=primary_features_512.unsqueeze(1))
        secondary_features, _ = self.multihead_attn(query=secondary_features_256.unsqueeze(1),
                                                    key=secondary_features_512.unsqueeze(1),
                                                    value=secondary_features_512.unsqueeze(1))

        if self.with_prism:
            wsi_latent = self.transform_prism(wsi_latent)
            wsi_latent = self.relu(wsi_latent)
            wsi_latent = self.drop(wsi_latent)

            wsi_latent = torch.stack([wsi_latent, wsi_latent, wsi_latent], dim=0)

            primary_features = torch.concat([primary_features, wsi_latent], dim=2)
            secondary_features = torch.concat([secondary_features, wsi_latent], dim=2)

        primary_logits = torch.empty(1, self.n_classes).to(primary_features.dtype).to(primary_features.device)
        for c in range(3):
            primary_logits[0, c] = self.classifiers_1[c](primary_features[c])

        secondary_logits = torch.empty(1, self.n_classes).to(secondary_features.dtype).to(secondary_features.device)
        for c in range(3):
            secondary_logits[0, c] = self.classifiers_2[c](secondary_features[c])

        primary_proba = self.logits_to_probas(primary_logits)
        secondary_proba = self.logits_to_probas(secondary_logits)

        # joint probability: P(Pri, Sec) = P(Pri) * P(Sec)
        joint_proba = torch.einsum('bi,bj->bij', primary_proba, secondary_proba)  # [B, 3, 3]

        # map to ISUP probability: P(ISUP=k) = sum_{Pri,Sec} P(Pri,Sec) * I(Pri,Sec->k)
        isup_proba = torch.einsum('bij,kij->bk', joint_proba, self.isup_matrix.to(x_input_256.device))

        primary_proba_ordinal = torch.flip(torch.flip(primary_proba, dims=[1]).cumsum(dim=1), dims=[1])[:, 1:]
        secondary_proba_ordinal = torch.flip(torch.flip(secondary_proba, dims=[1]).cumsum(dim=1), dims=[1])[:, 1:]
        isup_proba_ordinal = torch.flip(torch.flip(isup_proba, dims=[1]).cumsum(dim=1), dims=[1])[:, 1:]

        if self.captum:
            return primary_logits, A_primary_256, A_primary_512, primary_proba_ordinal, secondary_logits, A_secondary_256, A_secondary_512, secondary_proba_ordinal
        else:
            return primary_proba_ordinal, secondary_proba_ordinal, isup_proba_ordinal, isup_proba, total_inst_loss
