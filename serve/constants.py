# dirs
ASSETS_DIR = "assets"
LOG_DIR = "logs"

SUPPORT_CODE_LANGUAGES = ["Python", "Shell", "其他"]

# chat modes
CHAT_MODE = "聊天"
INFILL_MODE = "代码填充"

TITLE_DESC = "<p style='text-align: center; font-size: 50px; font-weight: bold;'>🐱 <span style='color: blue;'>C</span>ode<span style='color: blue;'>A</span>C<span style='color: blue;'>T</span></p>"
INTRO_DESC = "CodeACT-Coder是一款人性化的AI编程助手，致力于提升开发者的编程效率与体验。CAT具有智能语义理解能力，可以理解开发者的需求，提供智能的交互模式。"

# Chat model path
CHAT_BASE_MODEL_PATH = "/mnt/disk3/models/LLM/DeepSeek-Coder/6.7b-instruct" # "/mnt/disk3/models/LLM/CodeLlama-hf/7b-instruct"
CHAT_ADAPTER_MODEL_PATH = None
# Infilling model path
INFILL_BASE_MODEL_PATH = "/mnt/disk3/models/LLM/DeepSeek-Coder/6.7b-base" # "/mnt/disk3/models/LLM/CodeLlama-hf/7b" 
INFILL_ADAPTER_MODEL_PATH = None
# conversation template name
CONV_NAME = "cat-deepseek" #"cat-codellama"

FILL_TOKENS = {
    "cat-codellama": "<FILL_ME>",
    "cat-deepseek": "<｜fim▁hole｜>",
}