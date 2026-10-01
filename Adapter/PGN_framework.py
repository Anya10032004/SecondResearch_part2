class PGN_Framework(nn.module):
    def __init__(self. hidden_size, size_the_adapter, embedding_size):

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

            
# Class untuk  dibuat khusus untuk jadi tempat yang manggil Adapter(dari repo AdapterPGNBertOutput) 
# Class untuk menghubungkan si adapter dengan bert(dari repo AdapterPGNBertModel)
