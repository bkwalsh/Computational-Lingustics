import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttentionMLP(nn.Module):
    """ Attention MLP.
    """
    def __init__(self, input_size: int, hidden_size: int):
        """ Initializes the attention MLP for output token values.

        Parameters
        ----------
        input_size: Input size of the token embeddings
        hidden_size: Hidden size of the classifier
        """
        super(AttentionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """ Model inference.

        Parameters
        ----------
        token_embeddings: Token embeddings dim=[B, L, E]

        Returns
        -------
        pooled_representation: Attention weighted representation of token embeddings dim=[B, E]
        """
        attention_logits = self.mlp(token_embeddings)  # [B, L, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, L, 1]
        pooled_representation = torch.sum(attention_weights * token_embeddings, dim=1)  # [B, E]
        return pooled_representation


class Classifier(nn.Module):
    """ BERT/GPT2 Classifier.
    """
    def __init__(self, model_name: str, layers_to_unfreeze: int,
                 pooling: str, hidden_size: int, num_classes: int):
        """ Initializes the classifier.

        Parameters
        ----------
        model_name: Pretrained model name from HuggingFace
        layers_to_unfreeze: Number of pretrained layers to unfreeze
        pooling: 'CLS' (BERT only) or 'MLP'
        hidden_size: Hidden size of the classifier
        num_classes: Number of classes
        """
        super(Classifier, self).__init__()
        self.pooling = pooling
        # Load pretrained model
        self.model = AutoModel.from_pretrained(model_name)
        for name, param in self.model.named_parameters():
            if name.startswith('encoder.layer.'):
                layer_number = int(name.split('encoder.layer.')[-1].split('.')[0])
                if layer_number >= self.model.config.num_hidden_layers - layers_to_unfreeze:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif name.startswith('h.'):
                layer_number = int(name.split('h.')[-1].split('.')[0])
                if layer_number >= self.model.config.num_hidden_layers - layers_to_unfreeze:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif name in ['pooler.dense.bias', 'pooler.dense.weight'] and layers_to_unfreeze > 0:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if torch.cuda.is_available(): # use GPU if available
            self.model = self.model.cuda()
        # Initialize AttentionMLP
        if pooling == 'MLP':
            self.attention_mlp = AttentionMLP(self.model.config.hidden_size, hidden_size)
        # Initialize feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes))

    def forward(self, sentences: torch.Tensor) -> torch.Tensor:
        """ Model inference.

        Parameters
        ----------
        sentences: Tokenized sentence batch with 'input ids' and 'attention_mask'

        Returns
        -------
        logits: Logits of the model
        """
        features = self.model(**sentences)['last_hidden_state'] # [B, L, D]
        if self.pooling == 'CLS':
            pooled_representation = features[:, 0, :]
        elif self.pooling == 'MLP':
            pooled_representation = self.attention_mlp(features)
        logits = self.classifier(pooled_representation)
        return logits