import ollama

MODEL = "qwen2.5:7b"

def get_pod_count(namespace: str) -> str:
    """Return how many pods are running in a Kubernetes namespace.

    Args:
        namespace: the Kubernetes namespace to check
    """
    return f"{namespace} has 9 pods running"   # fake for now; real tools come in 5.2

msgs = [{"role": "user", "content": "How many pods are running in the boutique namespace?"}]

# Round 1: the model decides whether to call a tool, and with which arguments
resp = ollama.chat(model=MODEL, messages=msgs, tools=[get_pod_count])
print("1) model asked for:", resp.message.tool_calls)
msgs.append(resp.message)

# Our code runs the tool. The model never executes anything itself
for call in resp.message.tool_calls or []:
    result = get_pod_count(**call.function.arguments)
    print("2) tool returned:", result)
    msgs.append({"role": "tool", "content": result, "tool_name": call.function.name})

# Round 2: the model writes the answer from the tool result
final = ollama.chat(model=MODEL, messages=msgs)
print("3) final answer:", final.message.content)
