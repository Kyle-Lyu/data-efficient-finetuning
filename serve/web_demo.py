import os
import sys
os.environ["GRADIO_TEMP_DIR"] = "tmp/gradio"

CUR_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.dirname(CUR_DIR)))

import logging
import glob
import datetime
import argparse
import uuid
import json
from threading import Thread
import gradio as gr
from transformers import TextIteratorStreamer, StoppingCriteriaList

from serve.constants import (
    ASSETS_DIR,
    LOG_DIR,
    SUPPORT_CODE_LANGUAGES,
    CHAT_MODE,
    INFILL_MODE,
    TITLE_DESC,
    INTRO_DESC,
    CHAT_BASE_MODEL_PATH,
    CHAT_ADAPTER_MODEL_PATH,
    INFILL_BASE_MODEL_PATH,
    INFILL_ADAPTER_MODEL_PATH,
    CONV_NAME,
    FILL_TOKENS,
)
from serve.conversations import get_conv_template
from serve.utils import KeyWordsStoppingCriteria, load_model_tokenizer

os.makedirs(LOG_DIR, exist_ok=True)

# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create a file handler
file_handler = logging.FileHandler(os.path.join(CUR_DIR, LOG_DIR, "app.log"))
file_handler.setLevel(logging.DEBUG)
# create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

no_change_radio = gr.Radio()
enable_radio = gr.Radio(interactive=True)
disable_radio = gr.Radio(interactive=False)


class State:
    def __init__(self, conv_name:str):
        self.conv = get_conv_template(conv_name).copy()
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
    
    def to_gradio_chatbot(self):
        # return self.conv.to_gradio_chatbot()
        ret = []
        for i, (role, msg) in enumerate(self.conv.messages[self.conv.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    code, code_language = msg
                    msg = f"```{code_language.lower()}\n{code}\n```" if code_language != "其他" else code
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret
    
    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
            }
        )
        return base

    
def get_conv_log_filename():
    t = datetime.datetime.now()
    logfile_path = os.path.join(CUR_DIR, LOG_DIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.jsonl")
    return logfile_path


def get_icons(icons_path):
    icons = []
    for i, icon_path in enumerate(icons_path):
        icons.append((os.path.join(CUR_DIR, icon_path), f"icon {i+1}"))
    return icons


def set_chat_mode(chat_mode:str, chat_state:State, infill_state:State, code_language:str):
    if chat_mode == CHAT_MODE:
        state = chat_state
        if len(state.conv.messages[state.conv.offset:]) > 0:
            updated_btns = (enable_btn, enable_btn, no_change_btn)
        else:
            updated_btns = (disable_btn, disable_btn, no_change_btn)
        
        return (chat_state.to_gradio_chatbot(), gr.Textbox(visible=True), gr.Code(visible=False), gr.Dropdown(visible=False)) + updated_btns
    elif chat_mode == INFILL_MODE:
        state = infill_state
        if len(state.conv.messages[state.conv.offset:]) > 0:
            updated_btns = (enable_btn, enable_btn, no_change_btn)
        else:
            updated_btns = (disable_btn, disable_btn, no_change_btn)

        if code_language:
            if code_language == "其他":
                code_language = None
            else:
                code_language = code_language.lower()
            return (infill_state.to_gradio_chatbot(), gr.Textbox(visible=False), gr.Code(language=code_language, visible=True), gr.Dropdown(visible=True)) + updated_btns
    

def set_code_language(code_language:str):
    if code_language:
        if code_language == "其他":
            code_language = None
        else:
            code_language = code_language.lower()
        return gr.Code(language=code_language)
    

def record_icon_likes(icon_id):
    log_file = os.path.join(CUR_DIR, LOG_DIR, "icon_likes.jsonl")
    if not icon_id:
        gr.Info("未选择任何icon 👾 👽 👺 👻")
    else:
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            data = {
                "time": cur_time,
                "icon_id": icon_id,
                "like": 1,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        gr.Info("已提交，祝您生活愉快！ 💜")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat_base_model_path", type=str,
        default=CHAT_BASE_MODEL_PATH,
    )
    parser.add_argument(
        "--chat_lora_model_path", type=str,
        default=CHAT_ADAPTER_MODEL_PATH,
    )
    parser.add_argument(
        "--infill_base_model_path", type=str,
        default=INFILL_BASE_MODEL_PATH,
    )
    parser.add_argument(
        "--infill_lora_model_path", type=str,
        default=INFILL_ADAPTER_MODEL_PATH,
    )
    parser.add_argument(
        "--load_8bit", action="store_true",
    )
    parser.add_argument(
        "--load_4bit", action="store_true",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
    )
    parser.add_argument(
        "--conv_name",type=str,
        default=CONV_NAME,
    )
    parser.add_argument(
        "--host", type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port", type=int,
        default=7080,
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Whether to generate a public, shareable link",
    )

    args = parser.parse_args()
    return args


def regenerate(chat_mode:str, chat_state:State, infill_state:State):
    if chat_mode == CHAT_MODE:
        chat_state.conv.update_last_message(None)
        return (disable_radio, chat_state, infill_state, chat_state.to_gradio_chatbot(), "", gr.Code(), gr.Dropdown(interactive=False)) + (disable_btn,) * 3
    elif chat_mode == INFILL_MODE:
        infill_state.conv.update_last_message(None)
        return (disable_radio, chat_state, infill_state, infill_state.to_gradio_chatbot(), gr.Textbox(), "", gr.Dropdown(interactive=False)) + (disable_btn,) * 3


def clear_history(chat_mode:str, chat_state:State, infill_state:State):
    if chat_mode == CHAT_MODE:
        state = chat_state
        conv_name = state.conv.name
        return (State(conv_name), infill_state, [], "", gr.Code()) + (disable_btn,) * 2 + (enable_btn,)
    elif chat_mode == INFILL_MODE:
        state = infill_state
        conv_name = state.conv.name
        return (chat_state, State(conv_name), [], gr.Textbox(), "") + (disable_btn,) * 2 + (enable_btn,)
    

def model_stream_iter(params):
    chat_mode:str = params["chat_mode"]
    prompt:str = params["prompt"]
    conv_name:str = params["conv_name"]
    temperature:float = float(params.get("temperature", 0.3))
    top_p:float = float(params.get("top_p", 0.9))
    max_new_tokens:int = int(params.get("max_new_tokens", 2048))
    do_sample:bool = True if temperature > 0.001 else False
    stop_str = params.get("stop_str", None)
    stop_tokens_ids = params.get("stop_tokens_ids", None)

    # print info
    logger.info(f"\nGeneration Parameters:\ndo_sample={do_sample}\ttemperature={temperature}\ttop_p={top_p}\tmax_new_tokens={max_new_tokens}\tstop_str={stop_str}\tstop_tokens_ids={stop_tokens_ids}")
    logger.info(f"\nPrompt:\n{prompt}")

    if chat_mode == CHAT_MODE:
        input_ids = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device).input_ids
        if stop_str:
            if isinstance(stop_str, str):
                stop_str = [stop_str]
            stopping_criteria = StoppingCriteriaList(
                [KeyWordsStoppingCriteria(stop_str, chat_tokenizer, input_ids)],
            )
        else:
            stopping_criteria = None
        
        streamer = TextIteratorStreamer(
            chat_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=15,
        )
        thread = Thread(
            target=chat_model.generate,
            kwargs=dict(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
                use_cache=True,
                eos_token_id=chat_tokenizer.eos_token_id,
                bos_token_id=chat_tokenizer.bos_token_id,
                pad_token_id=chat_tokenizer.pad_token_id,
            )
        )
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield json.dumps({"text": generated_text, "error_code": 0}).encode()

    elif chat_mode == INFILL_MODE:
        filling_flag = False
        if "llama" in conv_name.lower():
            if "<FILL_ME>" in prompt:
                filling_flag = True
                full_prompt = prompt
        elif "deepseek" in conv_name.lower():
            if "<｜fim▁hole｜>" in prompt:
                filling_flag = True
                full_prompt = "<｜fim▁begin｜>" + prompt + "<｜fim▁end｜>"
        
        if filling_flag:
            input_ids = infill_tokenizer(full_prompt, return_tensors="pt").to(infill_model.device).input_ids
            outputs = infill_model.generate(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                eos_token_id=infill_tokenizer.pad_token_id,
                pad_token_id=infill_tokenizer.pad_token_id,
                stop_strings=stop_str,
                tokenizer=infill_tokenizer,
            )

            if "llama" in conv_name.lower():
                filling:str = infill_tokenizer.decode(outputs[0], skip_special_tokens=False)
                filling = filling.split("<MID>")[1]
                if stop_str:
                    if isinstance(stop_str, str):
                        stop_str = [stop_str]
                    for s in stop_str:
                        stop_index = filling.rfind(s)
                        if stop_index != -1:
                            filling = filling[:stop_index]
                            break
                full_text = prompt.replace("<FILL_ME>", filling)
            elif "deepseek" in conv_name.lower():
                filling:str = infill_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                full_text = prompt.replace("<｜fim▁hole｜>", filling)
        else:
            full_text = "检测到不支持的填充格式！"
        
        yield json.dumps({"text": full_text, "error_code": 0}).encode()
        return


def add_text(chat_mode:str, chat_state:State, infill_state:State, text:str, code:str, code_language:str):
    if chat_mode == CHAT_MODE:
        if len(text.strip()) <= 0:
            chat_state.skip_next = True
            return (no_change_radio, chat_state, infill_state, chat_state.to_gradio_chatbot(), "", gr.Code(), gr.Dropdown()) + (no_change_btn, ) * 3
        
        chat_state.conv.append_message(chat_state.conv.roles[0], text.strip())
        chat_state.conv.append_message(chat_state.conv.roles[1], None)
        chat_state.skip_next = False
        return (disable_radio, chat_state, infill_state, chat_state.to_gradio_chatbot(), "", gr.Code(), gr.Dropdown(interactive=False)) + (disable_btn,) * 3
    elif chat_mode == INFILL_MODE:
        if len(code.strip()) <= 0:
            infill_state.skip_next = True
            return (no_change_radio, chat_state, infill_state, infill_state.to_gradio_chatbot(), gr.Textbox(), "", gr.Dropdown()) + (no_change_btn, ) * 3
        
        infill_state.conv.append_message(infill_state.conv.roles[0], (code.strip(), code_language))
        infill_state.conv.append_message(infill_state.conv.roles[1], None)
        infill_state.skip_next = False
        return (disable_radio, chat_state, infill_state, infill_state.to_gradio_chatbot(), gr.Textbox, "", gr.Dropdown(interactive=False)) + (disable_btn,) * 3


def predict(chat_mode:str, chat_state:State, infill_state:State, temperature:float, top_p:float, max_new_tokens:int, system_prompt:str, code_language:str):
    if chat_mode == CHAT_MODE:
        state = chat_state
        if system_prompt:
            state.conv.set_system_message(system_prompt)
        prompt = state.conv.get_prompt()
    elif chat_mode == INFILL_MODE:
        state = infill_state
        prompt = infill_state.conv.messages[-2][-1][0]

    if state.skip_next:
        state.skip_next = False
        yield (no_change_radio, chat_state, infill_state, state.to_gradio_chatbot(), gr.Dropdown(interactive=True)) + (no_change_btn, ) * 3
        return

    gen_params = {
        "chat_mode": chat_mode,
        "prompt": prompt,
        "conv_name": state.conv.name,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "stop_str": state.conv.stop_str,
        "stop_tokens_ids": state.conv.stop_tokens_ids,
    }

    stream_iter = model_stream_iter(gen_params)

    # state.conv.update_last_message("▌")
    # yield (chat_state, infill_state, state.to_gradio_chatbot(), gr.Radio(interactive=False), gr.Dropdown(interactive=False)) + (disable_btn,) * 3

    try:
        for data in stream_iter:
            data = json.loads(data.decode())
            if data["error_code"] == 0:
                output = data["text"]
                state.conv.update_last_message(output + "▌")
                yield (disable_radio, chat_state, infill_state, state.to_gradio_chatbot(), gr.Dropdown(interactive=False)) + (disable_btn,) * 3
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                state.conv.update_last_message(output)
                yield (disable_radio, chat_state, infill_state, state.to_gradio_chatbot(), gr.Dropdown(interactive=False)) + (disable_btn,) * 3
        output:str = data["text"]
        if state.conv.stop_str:
            if isinstance(state.conv.stop_str, str):
                state.conv.stop_str = [state.conv.stop_str]
            for stop_str in state.conv.stop_str:
                stop_index = output.rfind(stop_str)
                if stop_index != -1:
                    output = output[:stop_index]
                    break
        output = output.strip()
        logger.info(f"\nResponse:\n{output}")

        if chat_mode == INFILL_MODE:
            output = f"```{code_language.lower()}\n{output}\n```" if code_language != "其他" else output
            
        state.conv.update_last_message(output)
        yield (enable_radio, chat_state, infill_state, state.to_gradio_chatbot(), gr.Dropdown(interactive=True)) + (enable_btn,) * 3
    except Exception as e:
        state.conv.update_last_message(
            f"error: {e}"
        )
        yield (enable_radio, chat_state, infill_state, state.to_gradio_chatbot(), gr.Dropdown(interactive=True)) + (enable_btn, ) * 3
        return
    
    with open(get_conv_log_filename(), "a", encoding="utf-8") as fout:
        if chat_mode == CHAT_MODE:
            chat_type = "chat"
        elif chat_mode == INFILL_MODE:
            chat_type = "infill"
        else:
            chat_type = "none"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "type": chat_type,
            "time": current_time,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "state": state.dict(),
        }
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")


def create_demo(conv_name:str):
    with gr.Blocks(
        theme="soft", title="CodeACT: 你的代码助手"
    ) as demo:
        chat_state = gr.State(value=State(conv_name))
        infill_state = gr.State(value=State(conv_name))
        
        # title and introduction content
        gr.HTML(TITLE_DESC)
        with gr.Accordion("简介", open=True):
            gr.Markdown(INTRO_DESC)

        with gr.Row():
            # set column for hyperparameters
            with gr.Column(scale=3):
                chat_mode = gr.Radio(
                    choices=[CHAT_MODE, INFILL_MODE],
                    value=CHAT_MODE,
                    label="模式",
                    interactive=True,
                )
                code_language = gr.Dropdown(
                    choices=SUPPORT_CODE_LANGUAGES,
                    value=SUPPORT_CODE_LANGUAGES[0],
                    label="编程语言",
                    show_label=True,
                    interactive=True,
                    visible=False,
                )

                with gr.Accordion("参数", open=True):
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                        visible=True,
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                        visible=True,
                    )
                    max_new_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=1024,
                        step=64,
                        interactive=True,
                        label="Max New Tokens"
                    )
                    system_prompt = gr.Textbox(
                        lines=2, 
                        label='System Prompt', 
                        placeholder="在此设置系统提示词", 
                    )

                with gr.Accordion("图标设计", open=False):
                    with gr.Row():
                        icons_path = sorted(glob.glob(os.path.join(CUR_DIR, ASSETS_DIR, "icon*.png")))
                        icons = get_icons(icons_path)
                        icons_gallery = gr.Gallery(
                            value=icons,
                            label="icons",
                            show_label=True,
                            container=True,
                            columns=2,
                            rows=2,
                        )
                    with gr.Row():
                        icon_names = gr.Dropdown(
                            choices=[caption for _, caption in icons],
                            value=None,
                            label="您最喜欢哪个icon？",
                            show_label=True,
                            visible=True,
                        )
                    with gr.Row():
                        icon_like_btn = gr.Button(value="🥰 喜欢这个icon", variant="primary")
            
            # main column for chat
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    label='CodeACT',
                    height=600,
                    show_copy_button=True,
                    avatar_images=[os.path.join(CUR_DIR, ASSETS_DIR, 'user.png'), os.path.join(CUR_DIR, ASSETS_DIR, 'icon.png')],
                )

                with gr.Row():
                    textbox = gr.Textbox(
                        lines=3, 
                        label='文本框', 
                        placeholder="在此输入", 
                        show_label=False, 
                        container=True,
                        interactive=True,
                        visible=True,
                    )
                    codebox = gr.Code(
                        lines=3,
                        label="代码框",
                        show_label=True,
                        container=True,
                        interactive=True,
                        visible=False,
                    )
                    
                with gr.Row():
                    clear_btn = gr.Button(value="🧹 清除历史", interactive=False)
                    regen_btn = gr.Button(value="🤔️ 重新生成", interactive=False)
                    query_btn = gr.Button(value="🚀 发送消息", variant="primary")


        # examples here
        gr.Examples(
            examples=[
                ['```cpp\nclass Rectangle {\n    public:\n        Rectangle(int w, int h) {\n            width = w;\n            height = h;\n        }\n\n        int area() {\n            return width * height;\n        }\n\n    private:\n        int width;\n        int height;\n};\n\nint main() {\n    Rectangle rect(5, 10);\n    cout << "Area: " << rect.area() << endl;\n    return 0;\n}\n```\n上述代码有问题吗？', None, CHAT_MODE],
                ['```java\npublic class AverageCalculator {\n    public static double calculateAverage(int[] numbers) {\n        int sum = 0;\n        for (int num : numbers) {\n            sum += num;\n        }\n        return (double) sum / numbers.length;\n    }\n}\n```\n为上述代码生成测试用例代码', None, CHAT_MODE],
                ["介绍Python中的列表推导式", None, CHAT_MODE],
                [None, f"def quick_sort(arr):\n    if len(arr) <= 1:\n         return arr\n    pivot = arr[0]\n    left = []\n    right = []\n{FILL_TOKENS[conv_name]}\n        if arr[i] < pivot:\n            left.append(arr[i])\n        else:\n            right.append(arr[i])\n    return quick_sort(left) + [pivot] + quick_sort(right)", INFILL_MODE],
                [None, f'def remove_non_ascii(s: str) -> str:\n    """ Remove non-ASCII characters from a string. """{FILL_TOKENS[conv_name]}\n    return result', INFILL_MODE],
            ],
            inputs=[textbox, codebox, chat_mode]
        )
                
        btn_list = [clear_btn, regen_btn, query_btn]

        chat_mode.change(
            set_chat_mode, 
            [chat_mode, chat_state, infill_state, code_language], 
            [chatbot, textbox, codebox, code_language] + btn_list,
        )
        code_language.change(
            set_code_language,
            [code_language],
            [codebox],
        )
        
        query_btn.click(
            add_text, 
            [chat_mode, chat_state, infill_state, textbox, codebox, code_language], 
            [chat_mode, chat_state, infill_state, chatbot, textbox, codebox, code_language] + btn_list,
        ).then(
            predict, 
            [chat_mode, chat_state, infill_state, temperature, top_p, max_new_tokens, system_prompt, code_language], 
            [chat_mode, chat_state, infill_state, chatbot, code_language] + btn_list,
        )
        regen_btn.click(
            regenerate,
            [chat_mode, chat_state, infill_state],
            [chat_mode, chat_state, infill_state, chatbot, textbox, codebox, code_language] + btn_list,
        ).then(
            predict, 
            [chat_mode, chat_state, infill_state, temperature, top_p, max_new_tokens, system_prompt, code_language], 
            [chat_mode, chat_state, infill_state, chatbot, code_language] + btn_list,
        )
        clear_btn.click(
            clear_history,
            [chat_mode, chat_state, infill_state],
            [chat_state, infill_state, chatbot, textbox, codebox] + btn_list,
        )
        # icon-likes button
        icon_like_btn.click(
            record_icon_likes,
            [icon_names],
            None,
        )
    
    return demo
        

if __name__ == "__main__":
    args = get_args()
    logger.info(args)

    device = f"cuda:{str(args.gpu)}" if args.gpu else None

    if not os.path.isdir(os.path.join(CUR_DIR, LOG_DIR)):
        os.makedirs(os.path.join(CUR_DIR, LOG_DIR), exist_ok=True)

    chat_model, chat_tokenizer = load_model_tokenizer(args.chat_base_model_path, args.chat_lora_model_path, args.conv_name, args.load_8bit, args.load_4bit, device)
    chat_model.eval()
    logger.info("Complete loading Chat Model")

    infill_model, infill_tokenizer = load_model_tokenizer(args.infill_base_model_path, args.infill_lora_model_path, args.conv_name, args.load_8bit, args.load_4bit, device)
    infill_model.eval()
    logger.info("Complete loading Infilling Model")

    demo = create_demo(args.conv_name)

    demo.queue(max_size=5)
    demo.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )
