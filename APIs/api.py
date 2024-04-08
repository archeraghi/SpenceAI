import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from mistralai import MistralClient

from utils.dataclass import GeneratingSpeechesRequest, RatingRequest
from utils.embedding import (
    calculate_similarity_mistral,
    calculate_similarity_openai,
    calculate_similarity_solar,
)
from utils.generate_speeches import generate_speeches
from utils.rate_statement import rate_statement

client = OpenAI(api_key="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",)
solar_client = OpenAI(
    api_key="hack-with-upstage-solar-0407", base_url="https://api.upstage.ai/v1/solar"
)
mitral_client = MistralClient(api_key="KkAs3Hi585hw802xxbTpjVxVwV1I5snx")


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/v1/generate-speeches")
def api_generate_arguments(request: GeneratingSpeechesRequest):
    speeches = generate_speeches(client, request.transcript)
    return speeches


@app.post("/v1/rate-statement")
def api_rate_statement(request: RatingRequest):
    rating = rate_statement(
        client,
        request.transcript,
        request.statement_from_mentor,
        request.statement_from_student,
    )

    similarity_mistral = calculate_similarity_mistral(
        mitral_client, request.statement_from_mentor, request.statement_from_student
    )
    similarity_solar = calculate_similarity_solar(
        solar_client, request.statement_from_mentor, request.statement_from_student
    )
    similarity_openai = calculate_similarity_openai(
        client, request.statement_from_mentor, request.statement_from_student
    )

    rating["similarity_mistral"] = similarity_mistral
    rating["similarity_solar"] = similarity_solar
    rating["similarity_openai"] = similarity_openai
    return rating


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
