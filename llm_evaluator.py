import json
from openai import OpenAI
from utils import get_api_key

eval_prompt = "You are a strict evaluation model designed for output verification. Your sole function is to determine if the Actual output is semantically similar to or logically consistent with the Expected output, given the Context information. Respond *only* with the single word 'True' or 'False', without any explanation, punctuation, or surrounding text. The Actual output must express the same core concept as the Expected output to be considered 'True'. Do not attempt to re-evaluate the correctness of the Expected output based on the Context."


def llm_evaluate(input, actual_output, expected_output, context, model_name="qwen/qwen3-vl-235b-a22b-instruct", eval_prompt=eval_prompt):
    base_url = "https://openrouter.ai/api/v1"

    api_key = get_api_key()

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": eval_prompt,
        },
        {
            "role": "user",
            "content": f"Input question: {input} \n Context information: {context} \n Expected output: {expected_output} \n"
                       f"Actual output: {actual_output}",
        }
    ]

    response = client.chat.completions.create(model=model_name, messages=messages)
    part_name = response.choices[0].message.content

    return part_name

if __name__ == "__main__":

    input = "What happens when the flashlight is on"
    actual_output = "NONE"
    expected_output = "The flashlight generates a source to use the magnifying glass to ignite the candle"
    context = "Analyze the image and identify the objects that are inside the blue game play area according to their properties. You are in the incredible machines 2 describe which object caused an outcome of the machine according to what's stated in TASK_DESCRIPTION under the list or NONE if not suitable and nothing else."

    print(llm_evaluate(input=input, actual_output=actual_output, expected_output=expected_output, context=context))
