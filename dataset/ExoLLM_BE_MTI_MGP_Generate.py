from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer
import torch
import pandas as pd
gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
data_name = 'BE'
exogenous_variables = ['Generation','System Load'] 
endogenous_variable = 'Belgium\'s Electricity Price'
data_collection = 'Belgium\'s electricity market'
exogenous_variables_text_template = 'Exogenous'
endogenous_template = 'Endogenous'
dataset_template1 = f'This dataset is {data_name}, containing the data collected from {data_collection}. Exogenous variables are'
dataset_template2 = f' and it is necessary to utilize these external variables sequentially to predict the endogenous variable'
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
    'Generation': [
        'This Exogenous variable is Generation, representing the total electricity generated within Belgium’s electricity market. The generation of electricity can vary depending on factors such as weather, generation capacity, and fuel availability, influencing the supply-demand balance and, consequently, electricity prices.',
        'Exogenous Generation reflects the electricity produced within Belgium. Fluctuations in generation, such as those caused by changes in renewable energy sources or conventional power plants, can significantly impact electricity prices by affecting the overall supply available in the market.',
        'Exogenous Generation indicates the amount of electricity generated in Belgium. When generation is high, it can lower electricity prices by increasing supply, while lower generation can drive prices up due to limited availability.',
        'Exogenous Generation captures the total electricity output in Belgium. Variations in generation, particularly from renewable sources, can have a direct effect on the supply-demand dynamics, influencing the overall electricity price in the market.'
    ],
    'System Load': [
        'This Exogenous variable is System Load, which represents the total electricity demand within Belgium. System load is influenced by factors such as time of day, seasonal trends, and weather conditions. Fluctuations in system load are closely tied to electricity prices, with higher demand typically leading to higher prices.',
        'Exogenous System Load reflects the overall demand for electricity across Belgium. During peak demand periods, system load can cause prices to rise due to the increased need for electricity to meet consumption levels.',
        'Exogenous System Load indicates the total demand for electricity in Belgium. Higher system load often leads to higher electricity prices, especially when generation struggles to meet the increased demand.',
        'Exogenous System Load captures the electricity consumption within Belgium. When system load is high, it signals increased demand, which can push up prices, especially if generation cannot fully meet that demand.'
    ],
    'Belgium\'s Electricity Price': [
        'This Endogenous variable is Belgium\'s Electricity Price, representing the price of electricity in Belgium’s electricity market. The price is directly influenced by generation levels and system load, with higher demand or lower supply typically leading to higher electricity prices.',
        'Endogenous Belgium\'s Electricity Price reflects the market price for electricity, which is shaped by the balance between generation and system load. When generation is insufficient to meet high demand, prices are likely to rise.',
        'Endogenous Belgium\'s Electricity Price indicates the price of electricity in Belgium. The price fluctuates based on the interaction between generation and system load, with higher loads or lower generation increasing the likelihood of price hikes.',
        'Endogenous Belgium\'s Electricity Price represents the price of electricity in the market, driven by both generation levels and system load. When generation is low or demand is high, electricity prices tend to rise due to supply-demand imbalances.'
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
torch.save(text_emds, data_name+'.pt')