# Vertex Model Garden Training Dataset Template

## Overview

Vertex Model Garden training provides templates for streamlined preprocessing of
datasets. Although datasets often have intricate structures, the supported LLM
models accept only flat strings. A template facilitates parsing a dataset and
preprocessing it to be compatible with the model.

When fine-tuning a pretrained model, it is advisable to maintain the same format
as the original training data. A template helps replicate the format, ensuring
consistency and potentially enhancing the fine-tuning process.

Both multi-turn messages and single instruction-response pairs are supported.
Multi-turn messages are accommodated using a more general `chat_template` field,
whereas simple instruction-response pair datasets are supported through the
`prompt_input` field.

A template is a JSON file consisting of string key-value pairs. Refer to the
following for the definitions of the supported fields.

## Template field documentation

**description**: An explanation of the template.

**source**: Information about the origin of the template.

**chat_template**: A
[jinja template](https://jinja.palletsprojects.com/en/3.1.x/templates/) that can
be used to parse a chat dataset. This is the same format as
[HF chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating).
To create a chat_template, use the `messages` variable to be filled with the
sample. The flag `--instruct_column_in_dataset` identifies which column will be
passed to the `messages` variable in the chat_template. This field is mutually
exclusive with `prompt_input` and `prompt_no_input`.

**prompt_input**: A string template that is used when value for the input column
exists in the sample. It should be able to be formatted with the
[str.format](https://docs.python.org/3/library/stdtypes.html#str.format) method.
The input column is specified with the flag `--instruct_column_in_dataset`. Used
for instruction dataset. This field is mutually exclusive with `chat_template`.

**prompt_no_input**: A string template that is used when value for the input
column does not exist in the sample. It should be able to be formatted with the
[str.format](https://docs.python.org/3/library/stdtypes.html#str.format) method.
The input column is specified with the flag `--instruct_column_in_dataset`. Used
for instruction dataset. This field is mutually exclusive with `chat_template`.

**instruction_separator**: A unique string used to indicate the start of the
instructions. If not specified, every token after response_separator will be
treated as a response, and every token before the first response_separator will
be treated as instruction.

**response_separator**: A unique string used to indicate the start of the
response. This field is required if `--completion_only` flag is set to `True`.

## Example templates

-   See the list of all supported templates [here](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/community-content/vertex_model_garden/model_oss/peft/train/vmg/templates).
-   For an example with `chat_template` see the JSON template below.

```
{
  "description": "Chat template used by Llama 3.",
  "source": "https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/a5a71a7527eac1d651bb145436c72026887fb68e/tokenizer_config.json#L2053",
  "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
  "instruction_separator": "<|start_header_id|>user<|end_header_id|>\n\n",
  "response_separator": "<|start_header_id|>assistant<|end_header_id|>\n\n"
}
```

-   For an example with `prompt_input` see the JSON template below. In this case
    the flag `--instruct_column_in_dataset=text` should be set, and there must
    be a column named `text` in the dataset.

```
{
  "description": "Template for openassistant-guanaco dataset.",
  "source": "https://huggingface.co/datasets/timdettmers/openassistant-guanaco",
  "prompt_input": "{text}",
  "instruction_separator": "### Human:",
  "response_separator": "### Assistant:"
}
```

-   For an example with `prompt_no_input` see the JSON template below. In this
    case the flag `--instruct_column_in_dataset=input` should be set, and there
    must be columns named `input` and `instruction` in the dataset.

```
{
  "description": "Template used by Alpaca-LoRA.",
  "source": "https://github.com/tloen/alpaca-lora/blob/main/templates/alpaca.json",
  "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
  "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
  "response_separator": "### Response:"
}
```
