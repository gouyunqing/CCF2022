from transformers import BertModel, AutoConfig
import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.config = AutoConfig.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.config.hidden_size, 36)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        # 获取[CLS]
        out = out[0][:, 0, :]
        out = self.dropout(out)
        outputs = self.fc(out)
        return outputs
