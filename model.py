import torch
import torch.nn as nn


class PersonalAI(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=128
    ):
        super().__init__()

        # Convert characters into vectors
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # Read the input question
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        # Generate the answer
        self.decoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        # Predict the next character
        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, question, answer_input):

        # Encode question
        question_embedded = self.embedding(question)

        _, hidden = self.encoder(question_embedded)

        # Decode answer
        answer_embedded = self.embedding(answer_input)

        output, _ = self.decoder(
            answer_embedded,
            hidden
        )

        # Predict characters
        predictions = self.output_layer(output)

        return predictions
