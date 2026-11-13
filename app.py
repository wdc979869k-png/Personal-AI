import streamlit as st
import torch

from model import PersonalAI


# -------------------------
# 1. Website Settings
# -------------------------

st.set_page_config(
    page_title="My Personal AI",
    page_icon="🤖"
)

st.title("My Personal AI")
st.write(
    "Ask me questions to learn more about me!"
)


# -------------------------
# 2. Load Trained Model
# -------------------------

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "model_weights.pth",
        map_location="cpu",
        weights_only=True
    )

    vocab = checkpoint["vocab"]

    model = PersonalAI(
        vocab_size=len(vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, vocab, checkpoint


model, vocab, checkpoint = load_model()

reverse_vocab = {
    value: key
    for key, value in vocab.items()
}


# -------------------------
# 3. Generate Answer
# -------------------------

def generate_answer(question):

    PAD = vocab["<PAD>"]
    SOS = vocab["<SOS>"]
    EOS = vocab["<EOS>"]

    max_q = checkpoint["max_q"]
    max_a = checkpoint["max_a"]

    question_ids = [
        vocab[char]
        for char in question
        if char in vocab
    ] + [EOS]

    question_ids = question_ids[:max_q]

    question_ids += (
        [PAD] * (max_q - len(question_ids))
    )

    question_tensor = torch.tensor(
        [question_ids],
        dtype=torch.long
    )

    answer_ids = [SOS]

    with torch.no_grad():

        question_embedded = model.embedding(
            question_tensor
        )

        _, hidden = model.encoder(
            question_embedded
        )

        for _ in range(max_a):

            input_tensor = torch.tensor(
                [[answer_ids[-1]]],
                dtype=torch.long
            )

            embedded = model.embedding(
                input_tensor
            )

            output, hidden = model.decoder(
                embedded,
                hidden
            )

            logits = model.output_layer(
                output[:, -1, :]
            )

            next_id = logits.argmax(
                dim=-1
            ).item()

            if next_id in [PAD, SOS, EOS]:
                break

            answer_ids.append(next_id)

    answer = "".join(
        reverse_vocab[token]
        for token in answer_ids[1:]
    )

    if not answer:
        return "I could not generate an answer."

    return answer


# -------------------------
# 4. Chat Interface
# -------------------------

question = st.text_input(
    "Enter your question:"
)

if st.button("Ask AI"):

    if question.strip():

        answer = generate_answer(question)

        st.subheader("AI Answer")

        st.write(answer)

    else:

        st.warning("Please enter a question.")
