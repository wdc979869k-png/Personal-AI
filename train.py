import pandas as pd
import torch
import torch.nn as nn

from model import PersonalAI


# -------------------------
# 1. Load Dataset
# -------------------------

data = pd.read_csv("training_data.csv")

questions = data["question"].astype(str).tolist()
answers = data["answer"].astype(str).tolist()

# Special tokens
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"

# Build character vocabulary
all_text = "".join(questions + answers)

characters = sorted(set(all_text))

vocab = {
    PAD: 0,
    SOS: 1,
    EOS: 2
}

for char in characters:
    if char not in vocab:
        vocab[char] = len(vocab)

print("Vocabulary size:", len(vocab))


# -------------------------
# 2. Convert Text to Numbers
# -------------------------

def encode(text):

    return (
        [vocab[char] for char in text]
        + [vocab[EOS]]
    )


encoded_questions = [
    encode(q) for q in questions
]

encoded_answers = [
    encode(a) for a in answers
]

max_q = max(len(x) for x in encoded_questions)
max_a = max(len(x) for x in encoded_answers)


def pad_sequence(sequence, length):

    sequence = sequence[:length]

    return sequence + [vocab[PAD]] * (
        length - len(sequence)
    )


# -------------------------
# 3. Initialize Model
# -------------------------

model = PersonalAI(
    vocab_size=len(vocab)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003
)

criterion = nn.CrossEntropyLoss(
    ignore_index=vocab[PAD]
)


# -------------------------
# 4. Training
# -------------------------

epochs = 200

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for question, answer in zip(
        encoded_questions,
        encoded_answers
    ):

        question_tensor = torch.tensor(
            [pad_sequence(question, max_q)],
            dtype=torch.long
        )

        # Decoder input starts with SOS
        answer_input = (
            [vocab[SOS]] + answer[:-1]
        )

        answer_target = answer

        answer_input_tensor = torch.tensor(
            [pad_sequence(answer_input, max_a)],
            dtype=torch.long
        )

        answer_target_tensor = torch.tensor(
            [pad_sequence(answer_target, max_a)],
            dtype=torch.long
        )

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(
            question_tensor,
            answer_input_tensor
        )

        # Calculate loss
        loss = criterion(
            predictions.reshape(-1, len(vocab)),
            answer_target_tensor.reshape(-1)
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(questions)

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {average_loss:.4f}"
        )


# -------------------------
# 5. Save Model
# -------------------------

torch.save(
    {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "max_q": max_q,
        "max_a": max_a,
        "embedding_dim": 64,
        "hidden_dim": 128
    },
    "model_weights.pth"
)

print("Training completed!")
print("Model weights saved!")
