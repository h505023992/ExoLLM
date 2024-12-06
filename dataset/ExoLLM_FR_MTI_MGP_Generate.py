from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer
import torch
import pandas as pd
gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
data_name = 'FR'
exogenous_variables = ['Generation','System Load'] 
endogenous_variable = 'France\'s Electricity Price'
data_collection = 'France\'s electricity market'
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
        'This Exogenous variable is Generation, representing the total electricity generated within France’s electricity market. The amount of generation can vary based on factors such as weather, energy policies, and the capacity of different power sources, which influences the overall supply in the market and can lead to fluctuations in electricity prices.',
        'Exogenous Generation reflects the total electricity produced in France. Changes in the generation mix, such as increases in renewable energy generation or decreases in fossil fuel power generation, can impact the supply-demand balance and, in turn, the electricity price.',
        'Exogenous Generation captures the electricity generation in France. Variations in generation levels, particularly from renewable sources, can have a direct impact on electricity prices, especially when demand exceeds supply.',
        'Exogenous Generation represents the total power generated within France. High generation levels generally lower electricity prices by increasing supply, while low generation levels can push prices up due to scarcity in supply.'
    ],
    'System Load': [
        'This Exogenous variable is System Load, indicating the total electricity demand within France. System load varies throughout the day and year, influenced by factors such as temperature, time of day, and overall economic activity. When system load is high, electricity prices often rise due to the increased demand for electricity.',
        'Exogenous System Load reflects the total demand for electricity in France. During peak demand periods, higher system load can drive up electricity prices, especially if generation is insufficient to meet this demand.',
        'Exogenous System Load represents the overall electricity consumption within France. When system load increases, it can place pressure on the supply, potentially leading to higher electricity prices, particularly during periods of high demand or low generation.',
        'Exogenous System Load captures the electricity consumption across France. A high system load indicates increased demand for electricity, which can drive electricity prices higher if the generation cannot keep up with the demand.'
    ],
    'France\'s Electricity Price': [
        'This Endogenous variable is France\'s Electricity Price, representing the price of electricity within the French market. The price is influenced by the interplay of generation and system load, with higher demand or lower supply typically resulting in higher electricity prices.',
        'Endogenous France\'s Electricity Price reflects the market price for electricity in France, determined by the balance between electricity generation and system load. When generation is insufficient to meet high demand, electricity prices are likely to rise.',
        'Endogenous France\'s Electricity Price indicates the price of electricity in the French market. This price fluctuates in response to the supply-demand balance, with higher loads or lower generation driving prices up.',
        'Endogenous France\'s Electricity Price represents the price of electricity in France’s market. When generation fails to meet demand, especially during peak load periods, the price of electricity increases due to the supply-demand imbalance.'
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