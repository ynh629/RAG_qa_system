#token控制
import tiktoken
def count_tokens(messages:list,model:str="qwen-plus") ->int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 4
    total = 0
    for msg in messages:
        total += tokens_per_message
        if isinstance(msg.get("content"), str):
            total += len(encoding.encode(msg["content"]))
        total += 2
    return total
def trim_message(messages:list,max_tokens:int = 3000,model:str="qwen-plus") -> list:
    system_msgs=[m for m in messages if m.get("role")=="system"]
    non_system_msgs=[m for m in messages if m.get("role")!="system"]
    if not non_system_msgs:
        return messages
    last_user_idx=None
    for i in range(len(non_system_msgs)-1,-1,-1):
        if non_system_msgs[i].get("role")=="user":
            last_user_idx=i
            break
    if last_user_idx is None:
        last_user_msg=[non_system_msgs[-1]]
        other_msgs=non_system_msgs[:-1]
    else:
        last_user_msg = [non_system_msgs[last_user_idx]]
        other_msgs = non_system_msgs[:last_user_idx] + non_system_msgs[last_user_idx + 1:]
    def build_from_others(others):
        return system_msgs + others + last_user_msg

    current=count_tokens(build_from_others(other_msgs),model)
    while current>max_tokens and len(other_msgs)>0:
        removed=other_msgs.pop(0)
        current=count_tokens(build_from_others(other_msgs),model)
    return build_from_others(other_msgs)

