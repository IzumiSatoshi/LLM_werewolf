from openai import OpenAI
openai_api_key = open("./openai_key.txt", "r").read().strip("\n")
client = OpenAI(api_key=openai_api_key)

completion = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {"role": "system" if False else "user", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "how many r in strawberry"
        }
    ]
)

print(completion.choices[0].message.content)