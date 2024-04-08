from sklearn.metrics.pairwise import cosine_similarity


def get_text_embedding_mistral(
    client, original_statement: str, student_statement: str,
):
    embeddings_batch_response = client.embeddings(
        model="mistral-embed", input=[original_statement, student_statement],
    )
    return embeddings_batch_response.data


def calculate_similarity_mistral(
    client, original_statement: str, student_statement: str
):
    embeddings = get_text_embedding_mistral(client, original_statement, student_statement)
    similarity = cosine_similarity([embeddings[0].embedding], [embeddings[1].embedding])
    similarity = similarity[0][0]
    return similarity


def calculate_similarity_solar(client, original_statement: str, student_statement: str):
    original_result = (
        client.embeddings.create(
            model="solar-1-mini-embedding-query", input=original_statement
        )
        .data[0]
        .embedding
    )

    student_result = (
        client.embeddings.create(
            model="solar-1-mini-embedding-passage", input=student_statement
        )
        .data[0]
        .embedding
    )

    similarity = cosine_similarity([original_result], [student_result])
    similarity = similarity[0][0]
    return similarity


def calculate_similarity_openai(
    client, original_statement: str, student_statement: str
):
    original_result = (
        client.embeddings.create(
            model="text-embedding-3-large", input=original_statement
        )
        .data[0]
        .embedding
    )

    student_result = (
        client.embeddings.create(
            model="text-embedding-3-large", input=student_statement
        )
        .data[0]
        .embedding
    )

    similarity = cosine_similarity([original_result], [student_result])
    similarity = similarity[0][0]
    return similarity
