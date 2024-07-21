import json
import os
from copy import deepcopy

from addict import Dict

from translation_modules import PDDL_Translator
from utils.pddl_output_utils import parse_new_predicates, parse_full_domain_model
from pddl_syntax_validator import PDDL_Syntax_Validator

from prompt_for_steps import ACTION_PARAMS_PROMPT_BASE, ACTION_PRECONDITIONS_PROMPT_BASE, ACTION_EFFECTS_PROMPT_BASE



def get_action_prompt(prompt_template, action_desc, include_extra_info):
    action_desc_prompt = action_desc['desc']
    if include_extra_info:
        for feedback_i in action_desc['extra_info']:
            action_desc_prompt += ' ' + feedback_i
    full_prompt = str(prompt_template) + ' ' + action_desc_prompt
    return full_prompt, action_desc_prompt

# SIMILAR FUNCTIONS FOR action_params_prompt, action_preconditions_prompt, action_effects_prompt

def get_action_params_prompt(action_desc , domain_desc_str, include_extra_info):
    action_params_prompt = ACTION_PARAMS_PROMPT_BASE + domain_desc_str
    action_desc_prompt = action_desc['desc']
    if include_extra_info:
        for feedback_i in action_desc['extra_info']:
            action_desc_prompt += ' ' + feedback_i
    full_prompt = str(action_params_prompt) + ' ' + action_desc_prompt
    return full_prompt, action_desc_prompt

def get_action_preconditions_prompt(action_desc , domain_desc_str, include_extra_info):
    action_preconditions_prompt = ACTION_PRECONDITIONS_PROMPT_BASE + domain_desc_str
    action_desc_prompt = action_desc['desc']
    if include_extra_info:
        for feedback_i in action_desc['extra_info']:
            action_desc_prompt += ' ' + feedback_i
    use_params_from_prev_step_str = """
    
    the parameters for the action are:
    The parameters that you defined in your previous answer.
    you can still add or remove parameters if you think that it is needed.
    
    """
    full_prompt = str(action_preconditions_prompt) + ' ' + action_desc_prompt + use_params_from_prev_step_str
    return full_prompt, action_desc_prompt

def get_action_effects_prompt(action_desc , domain_desc_str, include_extra_info):
    action_effects_prompt = ACTION_EFFECTS_PROMPT_BASE + domain_desc_str
    action_desc_prompt = action_desc['desc']
    if include_extra_info:
        for feedback_i in action_desc['extra_info']:
            action_desc_prompt += ' ' + feedback_i
    use_params_from_prev_step_str = """

    the parameters for the action are:
    The parameters that you defined in your previous answer.
    you can still add or remove parameters if you think that it is needed. 
    """

    use_preconditions_from_prev_step_str = """
    
    the preconditions for the action are:
    The preconditions that you defined in your previous answer.
    you can still add or remove preconditions if you think that it is needed. 
    """
    full_prompt = str(action_effects_prompt) + ' ' + action_desc_prompt + use_params_from_prev_step_str + use_preconditions_from_prev_step_str
    return full_prompt, action_desc_prompt

def get_predicate_prompt(predicate_list):
    predicate_prompt = 'You can create and define new predicates (please prefix each line with a number and a dot (1. , 2.), also please add a description for each new predicate, example: 1. (is-block ?b - blockObject): true if ?b is block ), but you may also reuse the following predicates:'
    if len(predicate_list) == 0:
        predicate_prompt += '\nNo predicate has been defined yet'
    else:
        for i, p in enumerate(predicate_list):
            predicate_prompt += f'\n{i+1}. {p["raw"]}'
    return predicate_prompt


def construct_action_model_in_3_steps(llm_conn, action_params_prompt, action_preconditions_prompt, action_effects_prompt, predicate_list):
    """
    Construct the action model in 3 steps: get parameters, get preconditions, get effects
    """
    # # print all the prompts
    # print(action_params_prompt)
    # print('***********' * 10)
    # print(action_preconditions_prompt)
    # print('***********' * 10)
    # print(action_effects_prompt)
    # raise
    conn_success , parameters_response = llm_conn.get_response(prompt=None, messages=[{'role': 'user', 'content': action_params_prompt}])
    predicate_list.extend(parse_new_predicates(parameters_response))

    conn_success, preconditions_response = llm_conn.get_response(prompt=None, messages=[{'role': 'user', 'content': action_params_prompt},
                                                                         {'role': 'assistant', 'content': parameters_response},
                                                                          {'role': 'user', 'content': action_preconditions_prompt}])
    predicate_list.extend(parse_new_predicates(preconditions_response))
    conn_success, effects_response = llm_conn.get_response(prompt=None, messages=[{'role': 'user', 'content': action_params_prompt},
                                                                    {'role': 'assistant', 'content': parameters_response},
                                                                    {'role': 'user', 'content': action_preconditions_prompt},
                                                                    {'role': 'assistant', 'content': preconditions_response},
                                                                    {'role': 'user', 'content': action_effects_prompt}])
    predicate_list.extend(parse_new_predicates(effects_response))
    messages = [{'role': 'user', 'content': action_params_prompt},
                {'role': 'assistant', 'content': parameters_response},
                {'role': 'user', 'content': action_preconditions_prompt},
                {'role': 'assistant', 'content': preconditions_response},
                {'role': 'user', 'content': action_effects_prompt},
                {'role': 'assistant', 'content': effects_response}]

    return effects_response, messages, predicate_list


def print_in_red(text):
    print(f'\033[91m{text}\033[0m')

def print_in_blue(text):
    print(f'\033[94m{text}\033[0m')
def construct_action_model(llm_conn, action_params_prompt, action_preconditions_prompt, action_effects_prompt,
                           action_name, predicate_list, max_iteration=3, end_when_error=False,
                           shorten_message=False, syntax_validator=None):
    """
    TODO: Should be changed to construct the action model by steps (get parameters, get preconditions, get effects)
    """
    def _shorten_message(_msg, _step_i):
        """
        Only keep the latest LLM output and correction feedback
        """
        try:
            print(f'[INFO] step: {_step_i} | num of messages: {len(_msg)}')
            if _step_i == 1:
                return [_msg[0]]
            else:
                _short_msg = [_msg[0], _msg[2 * (_step_i - 1) - 1], _msg[2 * (_step_i - 1)]]
                assert _short_msg[1]['role'] == 'assistant'
                assert _short_msg[2]['role'] == 'user'
                print_in_blue(f"was able to shorten the message ")
                return _short_msg
        except Exception as err:
            print_in_red(f'[ERROR] fail to shorten the message')
            return _msg

    results_dict = Dict({action_name: Dict()})

    effect_response, messages, predicate_list = construct_action_model_in_3_steps(llm_conn, action_params_prompt, action_preconditions_prompt, action_effects_prompt, predicate_list)
    conn_success, llm_output = False, messages[-1]['content']
    new_predicates = parse_new_predicates(llm_output)
    predicate_list.extend(new_predicates)
    results_dict[action_name]['new_predicates'] = [new_p['raw'] for new_p in new_predicates]

    ## Skipping the syntax validation for now

    return llm_output, results_dict, predicate_list


def main():
    actions = None  # None means all actions
    skip_actions = None
    prompt_version = 'model_blocksworld'
    include_additional_info = True
    # domain = 'tyreworld'  # 'household', 'logistics', 'tyreworld'
    domain = 'household'  # 'household', 'logistics', 'tyreworld'
    engine = 'gpt-4'  # 'gpt-4' or 'gpt-3.5-turbo'
    end_when_error = False      # whether to end the experiment when having connection error
    unsupported_keywords = ['forall', 'when', 'exists', 'implies']
    max_iterations = 3 if ('gpt-4' in engine and domain != 'household') else 2  # we only do 2 iteration in Household because there are too many actions, so the experiments are expensive to run
    max_feedback = 3   # more feedback doesn't help with other models like gpt-3.5-turbo
    shorten_messages = False
    claude_engine = 'claude-3-5-sonnet-20240620'    # Can be one of ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229' ,claude-3-haiku-20240307] # haiku should be the best, but super expensive
    use_claude = True
    experiment_name = 'by_steps'

    pddl_prompt_dir = f'prompts/common/'
    domain_data_dir = f'prompts/{domain}'
    with open(os.path.join(pddl_prompt_dir, f'{prompt_version}/pddl_prompt.txt')) as f:
        prompt_template = f.read().strip()
    with open(os.path.join(domain_data_dir, f'domain_desc.txt')) as f:
        domain_desc_str = f.read().strip()
        if '{domain_desc}' in prompt_template:
            prompt_template = prompt_template.replace('{domain_desc}', domain_desc_str)
    with open(os.path.join(domain_data_dir, f'action_model.json')) as f:
        action_desc = json.load(f)
    with open(os.path.join(domain_data_dir, f'hierarchy_requirements.json')) as f:
        obj_hierarchy_info = json.load(f)['hierarchy']

    # only GPT-4 is able to revise PDDL models with feedback message
    syntax_validator = PDDL_Syntax_Validator(obj_hierarchy_info, unsupported_keywords=unsupported_keywords) if 'gpt-4' in engine else None
    pddl_translator = PDDL_Translator(domain, engine='gpt-4')

    if actions is None:
        actions = list(action_desc.keys())
    predicate_list = list()

    # init LLM
    from llm_model import GPT_Chat
    llm_gpt = GPT_Chat(engine=engine)

    from llm_model import Anthropic_Chat
    llm_anthropic = Anthropic_Chat(engine=claude_engine)

    llm_obj = llm_anthropic if use_claude else llm_gpt

    results_dict = Dict()
    result_log_dir = f"{'antrhopic' if use_claude else 'gpt'}/results/{experiment_name}/{domain}/{prompt_version}"
    os.makedirs(result_log_dir, exist_ok=True)

    for i_iter in range(max_iterations):
        readable_results = ''
        prev_predicate_list = deepcopy(predicate_list)

        for i_action, action in enumerate(actions):
            if skip_actions is not None and action in skip_actions:
                continue


            action_params_prompt, action_desc_prompt = get_action_params_prompt(action_desc[action], domain_desc_str,
                                                                                include_additional_info)
            action_preconditions_prompt, action_desc_prompt = get_action_preconditions_prompt(action_desc[action], domain_desc_str,
                                                                                include_additional_info)
            action_effects_prompt, action_desc_prompt = get_action_effects_prompt(action_desc[action], domain_desc_str,
                                                                                include_additional_info)


            print('\n')
            print('#' * 20)
            print(f'[INFO] iter {i_iter} | action {i_action}: {action}.')
            print('#' * 20)
            readable_results += '\n' * 2 + '#' * 20 + '\n' + f'Action: {action}\n' + '#' * 20 + '\n'

            predicate_prompt = get_predicate_prompt(predicate_list) # Can stay as is
            results_dict[action]['predicate_prompt'] = predicate_prompt
            results_dict[action]['action_desc'] = action_desc_prompt
            readable_results += '-' * 20
            readable_results += f'\n{predicate_prompt}\n' + '-' * 20

            action_params_prompt = f'{action_params_prompt}\n\n{predicate_prompt}\n\nParameters:'
            action_preconditions_prompt = f'{action_preconditions_prompt}\n\n{predicate_prompt}\n\nParameters:'
            action_effects_prompt = f'{action_effects_prompt}\n\n{predicate_prompt}\n\nParameters:'


            pddl_construction_output = construct_action_model(llm_obj, action_params_prompt, action_preconditions_prompt, action_effects_prompt, action, predicate_list,
                                                              shorten_message=shorten_messages, max_iteration=max_feedback,
                                                              end_when_error=end_when_error, syntax_validator=syntax_validator)
            llm_output, action_results_dict, predicate_list = pddl_construction_output

            results_dict.update(action_results_dict)
            readable_results += '\n' + '-' * 10 + '-' * 10 + '\n'
            readable_results += llm_output + '\n'

        readable_results += '\n' + '-' * 10 + '-' * 10 + '\n'
        readable_results += 'Extracted predicates:\n'
        for i, p in enumerate(predicate_list):
            readable_results += f'\n{i + 1}. {p["raw"]}'

        with open(os.path.join(result_log_dir, f'{engine}_0_{i_iter}.txt'), 'w') as f:
            f.write(readable_results)
        with open(os.path.join(result_log_dir, f'{engine}_0_{i_iter}.json'), 'w') as f:
            json.dump(results_dict, f, indent=4, sort_keys=False)

        gen_done = False
        if len(prev_predicate_list) == len(predicate_list):
            print(f'[INFO] iter {i_iter} | no new predicate has been defined, will terminate the process')
            gen_done = True

        if gen_done:
            break

    # finalize the pddl model and translate it
    parsed_domain_model = parse_full_domain_model(results_dict, actions)
    translated_domain_model = pddl_translator.translate_domain_model(parsed_domain_model, predicate_list, action_desc)
    # save the result
    with open(os.path.join(result_log_dir, f'{engine}_pddl.json'), 'w') as f:
        json.dump(translated_domain_model, f, indent=4, sort_keys=False)
    # save the predicates
    predicate_list_str = ''
    for idx, predicate in enumerate(predicate_list):
        if idx == 0:
            predicate_list_str += predicate['raw']
        else:
            predicate_list_str += '\n' + predicate['raw']
    with open(os.path.join(result_log_dir, f'{engine}_predicates.txt'), 'w') as f:
        f.write(predicate_list_str)
    # save the expr log for annotation
    simplified_result_dict = Dict()
    for act in results_dict:
        if act not in action_desc:
            continue
        simplified_result_dict[act]['action_desc'] = results_dict[act]['action_desc']
        simplified_result_dict[act]['llm_output'] = results_dict[act]['llm_output']
        simplified_result_dict[act]['translated_preconditions'] = translated_domain_model[act]['translated_preconditions']
        simplified_result_dict[act]['translated_effects'] = translated_domain_model[act]['translated_effects']
        simplified_result_dict[act]['annotation'] = list()
    with open(os.path.join(result_log_dir, f'{engine}_pddl_for_annotations.json'), 'w') as f:
        json.dump(simplified_result_dict, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    main()
