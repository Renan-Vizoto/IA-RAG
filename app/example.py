from ollama import Client

client = Client(host='http://localhost:11435')

def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city
    """
    # your actual logic here
    return f"25°C and sunny in {city}"

available_tools = {'get_weather': get_weather}

messages = [{'role': 'user', 'content': 'What is the weather in Tokyo? And whats the capital of france?'}]

response = client.chat(
    model='qwen3.5:4b',
    messages=messages,
    tools=[get_weather],  # pass the function directly
    think=False
)

# Handle tool calls
for tool_call in response.message.tool_calls or []:
    fn = available_tools.get(tool_call.function.name)
    if fn:
        result = fn(**tool_call.function.arguments)
        
        # Send result back to model
        messages.append(response.message)
        print(response.message)
        messages.append({'role': 'tool', 'content': result})
        
        final = client.chat(model='qwen3.5:4b', messages=messages)
        print(final.message.content)