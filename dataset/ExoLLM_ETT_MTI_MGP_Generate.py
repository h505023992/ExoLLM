from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer
import torch
import pandas as pd
gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
data_name = 'ETTh1'
df = pd.read_csv(data_name+'.csv', header=0)
variables = df.columns[1:]
exogenous_variables = df.columns[1:-1]
exogenous_variables = ['High UseFul Load','High UseLess Load','Middle UseFul Load','Middle UseLess Load','Low UseFul Load','Low UseLess Load']
endogenous_variable = 'Oil Temperature'
exogenous_variables_text_template = 'Exogenous'
endogenous_template = 'Endogenous'
dataset_template1 = f'This dataset is {data_name}, containing the data collected from electricity transformers. Exogenous variables are'
dataset_template2 = f' and it is necessary to utilize these exogenous variables sequentially to predict the endogenous variable'
trend_template = ['series shows an overall upward trend',
                  'series initially rises and then declines',
                  'series exhibits an overall declining trend',
                  'series initially declines and then rises']
period_template = ['series has no apparent periodicity',
                   'series exhibits shorter periodicity and higher frequency',
                   'series displays clear periodicity',
                   'series exhibits relatively longer periodicity']
stability_template = ['series undergoes significant instability over all the time',
                      'series remains relatively stable with minimal fluctuations',
                      'series experiences occasional bouts of volatility, interspersed with periods of relative calm',
                      'series shows consistent stability, with values remaining close to a steady mean']
noise_template = ['series is subject to very strong noise interference',
                  'series has a low signal-to-noise ratio, where noise significantly affects the clarity of the underlying data',
                  'series experiences moderate noise, partially obscuring the underlying pattern',
                  'series is not influenced by any noise interference']
nature_attribute_template = {
    'High UseLess Load': [
        'This Exogenous variable is High UseLess Load, representing external load that is inefficiently utilized',
        'Exogenous High UseLess Load indicates a potential inefficiency in the system’s external load handling',
        'Exogenous High UseLess Load can lead to increased energy consumption without corresponding output',
        'Exogenous High UseLess Load might suggest that the system is operating under suboptimal external conditions'
    ],
    'High UseFul Load': [
        'This Exogenous variable is High UseFul Load, representing external load that is efficiently utilized',
        'Exogenous High UseFul Load indicates the system is leveraging external loads effectively for improved performance',
        'Exogenous High UseFul Load highlights an optimal alignment between external inputs and system output',
        'Exogenous High UseFul Load suggests that the system operates under favorable external conditions'
    ],
    'Middle UseFul Load': [
        'This Exogenous variable is Middle UseFul Load, indicating moderate external load utilization',
        'Exogenous Middle UseFul Load suggests partial efficiency in how the system handles external loads',
        'Exogenous Middle UseFul Load could signify a balance between energy use and system output',
        'Exogenous Middle UseFul Load may point to an average external influence on the system’s performance'
    ],
    'Middle UseLess Load': [
        'This Exogenous variable is Middle UseLess Load, signifying a moderate inefficiency in external load utilization',
        'Exogenous Middle UseLess Load indicates that some external loads may not be fully optimized',
        'Exogenous Middle UseLess Load can lead to occasional energy inefficiencies or reduced system output',
        'Exogenous Middle UseLess Load might reflect fluctuating external conditions affecting system performance'
    ],
    'Low UseFul Load': [
        'This Exogenous variable is Low UseFul Load, representing minimal but efficient external load utilization',
        'Exogenous Low UseFul Load suggests that a small external load contributes effectively to system performance',
        'Exogenous Low UseFul Load highlights the ability of the system to maintain efficiency under low external load conditions',
        'Exogenous Low UseFul Load might indicate stable operations with minimal external influence'
    ],
    'Low UseLess Load': [
        'This Exogenous variable is Low UseLess Load, representing minimal external load that is not efficiently utilized',
        'Exogenous Low UseLess Load indicates that inefficiencies exist even under low external load conditions',
        'Exogenous Low UseLess Load could suggest an opportunity to enhance system performance by minimizing waste',
        'Exogenous Low UseLess Load might reflect limited but suboptimal external interactions with the system'
    ],
    'Oil Temperature':[
    'This Endogenous variable is Oil Temperature, representing the temperature of the oil within the system',
    'Endogenous Oil Temperature indicates the thermal condition of the oil, which can impact system efficiency and performance',
    'Endogenous Oil Temperature could suggest potential overheating or optimal thermal conditions depending on its value',
    'Endogenous Oil Temperature reflects the thermal stability and operational health of the system components'
    ]

}
prompts = []
for v in exogenous_variables:
    for nature_attribute in nature_attribute_template[v]:
        prompts.append(nature_attribute)
    for trend in trend_template:
        prompts.append(exogenous_variables_text_template+' '+ v+' '+ trend + '.')
    for period in period_template:
        prompts.append(exogenous_variables_text_template+' '+ v+' '+ period + '.')
    for stability in stability_template:
        prompts.append(exogenous_variables_text_template+' '+ v+' '+ stability + '.')
    for noise in noise_template:
        prompts.append(exogenous_variables_text_template+' '+ v+' '+ noise + '.')

    dataset_template1 = dataset_template1+' '+v+','
for nature_attribute in nature_attribute_template[endogenous_variable]:
    prompts.append(nature_attribute)
for trend in trend_template:
    prompts.append(endogenous_template+' '+ endogenous_variable+' '+ trend + '.')
for period in period_template:
    prompts.append(endogenous_template+' '+ endogenous_variable+' '+ period + '.')
for stability in stability_template:
    prompts.append(endogenous_template+' '+ endogenous_variable+' '+ stability + '.')
for noise in noise_template:
    prompts.append(endogenous_template+' '+ endogenous_variable+' '+ noise + '.')

dataset_template2 = dataset_template2 + ' ' +  endogenous_variable+ '.'
dataset_template1 = dataset_template1 + dataset_template2
prompts.append(dataset_template1)

text_emds = []
eos_token = "<|endoftext|>" 
for p in prompts:
    prompts_with_eos = p + eos_token
    tokens = tokenizer(prompts_with_eos, padding=False, truncation=True, return_tensors="pt")
    input_ids = tokens["input_ids"]
    with torch.no_grad():
        outputs = gpt2(input_ids=input_ids)
    text_emds.append(outputs.last_hidden_state[:,-1])
text_emds = torch.cat(text_emds,dim=0)
for data_name in ['ETTh1','ETTm1','ETTh2','ETTm2']:
    torch.save(text_emds, data_name+'.pt')