import json

from openai import OpenAI


def rate_statement(
    client: OpenAI,
    court_transcript: str,
    original_statement: str,
    statement_from_student: str,
):
    resp = client.chat.completions.create(
        response_format={"type": "json_object"},
        seed=42,
        temperature=0.0,
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a world-class AI legal mentor, equipped to evaluate the quality of a student's statement in comparison to an original statement based on a court transcript.",
                        "Your objective is to provide a rating that reflects the quality of the student's statement in relation to the original statement.",
                        "",
                        "The process includes:",
                        "1. Carefully analyzing the original statement and the student's statement.",
                        "2. Comparing the content and relevance of the two statements.",
                        "3. Do not focus on the details of the court transcript, but rather on the quality of the statements.",
                        "4. Assigning a rating based on the student's statement, with a focus on accuracy and coherence.",
                        "5. Structuring the output in JSON format, adhering to the template: {'feedback': str, 'rating': int}.",
                        "6. The rating should be between 1 and 3, with 1 indicating a poor statement, 2 indicating a mediocre statement, and 3 indicating an excellent statement.",
                        "7. The more semantically similar the student's statement is to the original statement, the higher the rating should be.",
                        "8. Even if the student's statement is not semantically similar to the original statement, it can still receive a high rating if it makes sense in the context of the court transcript.",
                        "9. The feedback should provide constructive criticism and highlight areas for improvement.",
                        "10. You should be able to generate the feedback even though the original statement is not provided in the prompt.",
                        "",
                        "Your task encompasses the following steps:",
                        "- Read and analyze the original statement and the student's statement.",
                        "- Compare the two statements based on content, structure, and relevance.",
                        "- Provide a rating between 1 and 3, along with constructive feedback for the student.",
                        "",
                        f"Original statement: {original_statement}",
                        f"Student's statement: {statement_from_student}",
                        f"Court Transcript: {court_transcript}",
                        "",
                        "Please proceed to rate the student's statement based on the guidelines and the original statement provided.",
                    ]
                ),
            }
        ],
    )

    rating = json.loads(resp.choices[0].message.content)
    cost = (
        resp.usage.completion_tokens * 0.01 + resp.usage.prompt_tokens * 0.03
    ) / 1000
    print(cost)

    return rating
