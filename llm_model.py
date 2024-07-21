import os
import traceback
from anthropic import Anthropic
import openai
from openai import OpenAI
openai_api_key = ""
entrhopic_api_key = ""

def connect_openai(client, engine, messages, temperature, max_tokens,
                   top_p, frequency_penalty, presence_penalty, stop):
    return client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )


class GPT_Chat:
    def __init__(self, engine, stop=None, max_tokens=1000, temperature=0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0):
        self.engine = engine
        self.gpt_client = OpenAI(api_key=openai_api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def get_response(self, prompt, messages=None, end_when_error=False, max_retry=5):
        conn_success, llm_output = False, ''
        if messages is not None:
            messages = messages
        else:
            messages = [{'role': 'user', 'content': prompt}]
        n_retry = 0
        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            try:
                print('[INFO] connecting to the LLM ...')
                response = connect_openai(
                    client=self.gpt_client,
                    engine=self.engine,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.freq_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop
                )
                llm_output = response.choices[0].message.content
                conn_success = True
            except Exception as e:
                print(f'[ERROR] LLM error: {e}')
                print(traceback.format_exc())
                if end_when_error:
                    break
        return conn_success, llm_output


class Anthropic_Chat:

    def __init__(self, engine, stop=None, max_tokens=1000, temperature=0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0):
        self.engine = engine
        self.anthropic_client = Anthropic(api_key=entrhopic_api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def get_response(self, prompt, messages=None, end_when_error=False, max_retry=5):
        conn_success, llm_output = False, ''
        if messages is not None:
            messages = messages
        else:
            messages = [{'role': 'user', 'content': prompt}]
        n_retry = 0
        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            try:
                print('[INFO] connecting to the Anthropic LLM ...')
                response = self.anthropic_client.messages.create(
                    model=self.engine,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                llm_output = response.content[0].text
                conn_success = True
            except Exception as e:
                print(f'[ERROR] LLM error: {e}')
                print(traceback.format_exc())
                if end_when_error:
                    break
        return conn_success, llm_output



def main():
    llm_gpt = GPT_Chat(engine='gpt-4')
    _, llm_output = llm_gpt.get_response('test', end_when_error=True)
    print(llm_output)


if __name__ == '__main__':
    main()
