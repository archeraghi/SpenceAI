import json

from openai import OpenAI


def generate_speeches(
    client: OpenAI, court_transcript: str,
):
    resp = client.chat.completions.create(
        response_format={"type": "json_object"},
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are an world-class AI legal mentor, equipped to create courtroom speeches based on a court transcript.",
                        "Your objective is to craft speeches for educational purposes, enabling students to practice and refine their argumentation skills in a legal context.",
                        "",
                        "The process includes:",
                        "1. Careful analysis of the provided court transcript.",
                        "2. Generate the speeches which would be provided as next conversation.",
                        "2. Generating up to five speeches that resonate with the content and style.",
                        "3. The speeches should be detailed but slightly different from the original argument and each other.",
                        "4. Structuring the output in JSON format, adhering to the template: {'speeches': ['speech1', 'speech2', ...]}.",
                        "5. Do not include any greetings in the output.",
                        "",
                        "Your task encompasses the following steps:"
                        "- Read and interpret the court transcript provided.",
                        "- Create speeches that are of a similar length and complexity, ensuring they are rooted in the details provided in the transcript.",
                        "- Format the output as specified, with a cap of five speeches.",
                        "",
                        f"Court Transcript: {court_transcript}",
                        "",
                        "Please proceed to generate the speeches based on the guidelines and the court transcript provided.",
                    ]
                ),
            }
        ],
    )

    speeches = json.loads(resp.choices[0].message.content)
    cost = (
        resp.usage.completion_tokens * 0.0005 + resp.usage.prompt_tokens * 0.0015
    ) / 1000
    print(cost)

    return speeches
