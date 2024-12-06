from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer
import torch
import pandas as pd
gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
data_name = 'PJM'
exogenous_variables = ['System Load','SyZonal COMED load'] 
endogenous_variable = 'Pennsylvania-New Jersey-Maryland Electricity Price'
data_collection = 'Pennsylvania-New Jersey-Maryland market'
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
    'System Load': [
        'This Exogenous variable is System Load, representing the total electricity demand across the entire system. System load reflects the overall consumption patterns, which are influenced by factors such as time of day, weather conditions, and seasonal changes. Fluctuations in system load are critical for understanding electricity price movements in the market.',
        'Exogenous System Load captures the demand for electricity within the grid. It plays a significant role in determining the market price of electricity, as higher loads typically lead to price increases due to greater demand on the system.',
        'Exogenous System Load indicates the total electricity consumption at any given time within the system. Variations in system load, especially during peak demand periods, can drive electricity prices up as the system works to meet higher consumption.',
        'Exogenous System Load reflects the overall electricity demand on the system. Significant fluctuations in load can impact electricity prices, with higher system loads often driving prices up, especially when the system is under stress.'
    ],
    'SyZonal COMED load': [
        'This Exogenous variable is SyZonal COMED load, representing the electricity consumption within the COMED service area, a part of the larger PJM grid. The consumption patterns in this region can significantly influence the broader market, as fluctuations in regional demand affect overall system supply-demand dynamics.',
        'Exogenous SyZonal COMED load reflects the electricity demand within the COMED zone of the PJM market. Changes in this load can lead to price shifts in the broader market, as local load patterns influence system-wide balancing and pricing.',
        'Exogenous SyZonal COMED load describes the electricity consumption in the COMED region, impacting the overall PJM system load and, subsequently, the electricity prices. A surge in this regional demand could cause price increases across the entire PJM market.',
        'Exogenous SyZonal COMED load indicates the electricity demand within the COMED service area, which is part of the larger PJM grid. Variations in demand within this zone may have ripple effects, influencing electricity prices across the PJM market.'
    ],
    'Pennsylvania-New Jersey-Maryland Electricity Price': [
        'This Endogenous variable is Pennsylvania-New Jersey-Maryland Electricity Price, representing the electricity price within the PJM market, specifically for the Pennsylvania-New Jersey-Maryland region. The price is influenced by a variety of factors, including system load, regional consumption patterns, and available generation resources.',
        'Endogenous Pennsylvania-New Jersey-Maryland Electricity Price reflects the market price of electricity in the PJM region, which is directly impacted by the interplay between system load, regional demand, and generation capacity. Price fluctuations reflect supply-demand imbalances.',
        'Endogenous Pennsylvania-New Jersey-Maryland Electricity Price represents the electricity price within the PJM market, influenced by both local demand factors like SyZonal COMED load and system-wide factors like System Load. Prices rise when demand exceeds supply within the region.',
        'Endogenous Pennsylvania-New Jersey-Maryland Electricity Price reflects the market price of electricity in the PJM region. This price is shaped by various factors such as system load and regional consumption patterns, with significant price volatility during periods of high demand or low supply.'
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
