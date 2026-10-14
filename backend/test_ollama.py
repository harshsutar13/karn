from ollama import chat

response = chat(
    model="gemma4:12b",
    messages=[
        {
            "role": "user",
            "content": "Explain predictive maintenance in an oil refinery in 3 sentences."
        }
    ],
)

print(response.message.content)