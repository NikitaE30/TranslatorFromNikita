import torch
import torch.nn as nn
import torch.optim as optim
from TranslatorFromNikita.trainer_models import Trainer as BaseTrainer
from .vocab import Vocab
import torch.nn.functional as F

class Tokenizer(nn.Module):
    def __init__(self, args, nchars, emb_dim, hidden_dim, _=5, dropout=0):
        super().__init__()
        self.args = args
        feat_dim = args["feat_dim"]
        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + feat_dim, hidden_dim, num_layers=self.args["rnn_layers"], bidirectional=True, batch_first=True, dropout=dropout if self.args["rnn_layers"] > 1 else 0)
        if self.args["conv_res"] is not None:
            self.conv_res = nn.ModuleList()
            self.conv_sizes = [int(x) for x in self.args["conv_res"].split(",")]
            for si, size in enumerate(self.conv_sizes):
                l = nn.Conv1d(emb_dim + feat_dim, hidden_dim * 2, size, padding=size//2, bias=self.args.get("hier_conv_res", False) or (si == 0))
                self.conv_res.append(l)
            if self.args.get("hier_conv_res", False):
                self.conv_res2 = nn.Conv1d(hidden_dim * 2 * len(self.conv_sizes), hidden_dim * 2, 1)
        self.tok_clf = nn.Linear(hidden_dim * 2, 1)
        self.sent_clf = nn.Linear(hidden_dim * 2, 1)
        self.mwt_clf = nn.Linear(hidden_dim * 2, 1)
        if args["hierarchical"]:
            in_dim = hidden_dim * 2
            self.rnn2 = nn.LSTM(in_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.tok_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.sent_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.mwt_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.toknoise = nn.Dropout(self.args["tok_noise"])

    def forward(self, x, feats):
        emb = self.embeddings(x)
        emb = self.dropout(emb)
        emb = torch.cat([emb, feats], 2)
        inp, _ = self.rnn(emb)
        if self.args["conv_res"] is not None:
            conv_input = emb.transpose(1, 2).contiguous()
            if not self.args.get("hier_conv_res", False):
                for l in self.conv_res:
                    inp = inp + l(conv_input).transpose(1, 2).contiguous()
            else:
                hid = []
                for l in self.conv_res:
                    hid += [l(conv_input)]
                hid = torch.cat(hid, 1)
                hid = F.relu(hid)
                hid = self.dropout(hid)
                inp = inp + self.conv_res2(hid).transpose(1, 2).contiguous()
        inp = self.dropout(inp)
        tok0 = self.tok_clf(inp)
        sent0 = self.sent_clf(inp)
        mwt0 = self.mwt_clf(inp)
        if self.args["hierarchical"]:
            if self.args["hier_invtemp"] > 0:
                inp2, _ = self.rnn2(inp * (1 - self.toknoise(torch.sigmoid(-tok0 * self.args["hier_invtemp"]))))
            else:
                inp2, _ = self.rnn2(inp)
            inp2 = self.dropout(inp2)
            tok0 = tok0 + self.tok_clf2(inp2)
            sent0 = sent0 + self.sent_clf2(inp2)
            mwt0 = mwt0 + self.mwt_clf2(inp2)
        nontok = F.logsigmoid(-tok0)
        tok = F.logsigmoid(tok0)
        nonsent = F.logsigmoid(-sent0)
        sent = F.logsigmoid(sent0)
        nonmwt = F.logsigmoid(-mwt0)
        mwt = F.logsigmoid(mwt0)
        pred = torch.cat([nontok, tok+nonsent+nonmwt, tok+sent+nonmwt, tok+nonsent+mwt, tok+sent+mwt], 2)
        return pred

class Trainer(BaseTrainer):
    def __init__(self, args=None, vocab=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            self.load(model_file)
        else:
            self.args = args
            self.vocab = vocab
            self.model = Tokenizer(self.args, self.args["vocab_size"], self.args["emb_dim"], self.args["hidden_dim"], dropout=self.args["dropout"])
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        if use_cuda:
            self.model.cuda()
            self.criterion.cuda()
        else:
            self.model.cpu()
            self.criterion.cpu()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(self.parameters, lr=self.args["lr0"], betas=(.9, .9), weight_decay=self.args["weight_decay"])
        self.feat_funcs = self.args.get("feat_funcs", None)
        self.lang = self.args["lang"]

    def update(self, inputs):
        self.model.train()
        units, labels, features, _ = inputs
        if self.use_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()
        pred = self.model(units, features)
        self.optimizer.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
        self.optimizer.step()
        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        units, labels, features, _ = inputs
        if self.use_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()
        pred = self.model(units, features)
        return pred.data.cpu().numpy()

    def save(self, filename):
        params = {
                "model": self.model.state_dict() if self.model is not None else None,
                "vocab": self.vocab.state_dict(),
                "config": self.args
                }
        try:
            torch.save(params, filename)
        except BaseException:
            ...

    def load(self, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            raise
        self.args = checkpoint["config"]
        self.model = Tokenizer(self.args, self.args["vocab_size"], self.args["emb_dim"], self.args["hidden_dim"], dropout=self.args["dropout"])
        self.model.load_state_dict(checkpoint["model"])
        self.vocab = Vocab.load_state_dict(checkpoint["vocab"])