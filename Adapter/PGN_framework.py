class PGN_Framework(nn.module): # nn.module is a class from pyTorch to make a model or neural network layer 
                                # so we add nn.module to inherit that class to the  PGN_Framework class
    def __init__(self. hidden_size, size_the_adapter, embedding_size):
        super(PGN_Framework, self).__init__()
        # hidden_size --> BERT's representation size.
        # size_the_adapter --> the smaller representaion inside bert
        # embedding_size --> When one word  turn to number, it will be represent how mcuh number in one vector 
        # self.config = config
        low_rank_dim = embedding_size

        # Prepare for The Weighth(down_W) and Bias(down_b) to shrink the Bert(768 --> 64)
        self.down_Weight = nn.Parameter(torch.zeros(low_rank_dim, hidden_size, size_the_adapter))
        self.down_bias = nn.Parameter(torch.zeros(low_rank_dim, size_the_adapter))

        # activattion function
        self.activation = nn.GELU()

        # Prepare The Weighth(up_W) and Bias(up_b) to increase back the Bert(64 --> 768)
        self.up_Weight = nn.Parameter(torch.zeros(low_rank_dim, hidden_size, size_the_adapter))
        self.up_bias = nn.Parameter(torch.zeros(low_rank_dim, size_the_adapter))
        self.init_weights()
        self.hidden_stage = hidden_size
        self.lang_emb = embedding_size and embedding_size != None or None

        def Down_and_Upsize_stage(self):
            # creating the  parameter for shrink the numbers, which are The Weighth(down_W) and Bias(down_b)
            down_Weight = torch.matmul(self.lang_emb, self.down_Weight.view(self.config.low_rank_dim, -1)).view(self.config.hidden_size, self.config.size_the_adapter)
            down_bias = torch.matmul(self.lang_emb, self.down_bias)
            down_projected = F.linear(self.hidden_stage, down_Weight.t(), down_bias)

            activated = self.activation(down_projected)

            up_Weight = torch.matmul(self.lang_emb, self.up_W.view(self.config.low_rank_dim, -1)).view(self.config.size_the_adapter, self.config.hidden_size)
            up_bias = torch.matmul(self.lang_emb, self.up_bias)
            up_projected = F.linear(activated, up_Weight.t(), up_bias)
            
            return self.hidden_stag + up_projected

        def init_weights(self):
            self.down_Weight.data.normal_(mean=0.0, std=0.0001)
            self.up_Weight.data.normal_(mean=0.0, std=0.0001)

class Bert_that_uses_Adapter(nn.module):
    def __init__(self, base, adapter_forward):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor, lang_emb=None):
        # Because Adaptor is a Neural network we will set 
        hidden_states = self.base.dense(hidden_states) 
        hidden_states = self.base.dropout(hidden_states) # 
        hidden_states = self.adapter_forward(hidden_states) # This is where start using the adapter
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class AdapterPGNBertModel(nn.Module):
    def __init__(self,
                 name_or_path_or_model: Union[str, BertModel],
                 adapter_size: int = 128,
                 language_emb_size: int = 32,
                 external_param: Union[bool, List[bool]] = False):
        super().__init__()
        if isinstance(name_or_path_or_model, str):
            self.bert = BertModel.from_pretrained(name_or_path_or_model)
        else:
            self.bert = name_or_path_or_model

        set_requires_grad(self.bert, False)

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(
                self.bert.config.num_hidden_layers)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(
                self.bert.config.num_hidden_layers)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e
        else:
            raise ValueError("wrong type of external_param!")

        self.adapters = nn.ModuleList([nn.ModuleList([
                AdapterWithParameterGen(self.bert.config.hidden_size, language_emb_size, adapter_size),
                AdapterWithParameterGen(self.bert.config.hidden_size, language_emb_size, adapter_size)
            ]) for e in param_place
        ])

        for i, layer in enumerate(self.bert.encoder.layer):
            layer.output = AdapterBertOutput(
                layer.output, self.adapters[i][0].forward)
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterBertOutput(
                layer.attention.output, self.adapters[i][1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        self.output_dim = self.bert.config.hidden_size

        # if word_piece == 'first':
        #     self.word_piece = None
        # else:  # mean of pieces
        #     offset = torch.tensor([0], dtype=torch.long)
        #     self.word_piece = lambda x: embedding_bag(
        #         x, self.bert.embeddings.word_embeddings.weight, offset.to(x.device))

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.LongTensor = None,
                mask: torch.Tensor = None,
                bert_pieces: torch.LongTensor = None
                ) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        # if self.word_piece is not None and word_pieces is not None:
        #     for (s, w), pieces in word_pieces.items():
        #         inputs_embeds[s, w, :] = self.word_piece(pieces)

        attention_mask = None if mask is None else mask.float()
        bert_output = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds, token_type_ids=token_type_ids)
        output = torch.bmm(bert_pieces, bert_output[self.bert_layers - 1])
        # return bert_output[0]
        return output

# Class untuk  dibuat khusus untuk jadi tempat yang manggil Adapter(dari repo AdapterPGNBertOutput) OKK
# Class untuk menghubungkan si adapter dengan bert(dari repo AdapterPGNBertModel);
#   Pastikan dicocokan kan dengan class lain
