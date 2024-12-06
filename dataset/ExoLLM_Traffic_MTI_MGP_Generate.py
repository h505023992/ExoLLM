from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer
import torch
import pandas as pd
gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
data_name = 'traffic'

def clean_text(text):
    cleaned_text = ''.join([char if ord(char) < 128 else '' for char in text])
    return cleaned_text

df = pd.read_csv(data_name+'.csv', header=0)

variables = df.columns[1:].tolist()
for i in range(len(variables )):
    variables[i]  = 'sensor '+clean_text(variables [i])
exogenous_variables = variables[:-1]
endogenous_variable = 'sensor 861'
exogenous_variables_text_template = 'Exogenous'
endogenous_template = 'Endogenous'
dataset_template1 = f'This dataset is {data_name}, containing hourly data from California Department of Transportation, which describes the road occupancy rates measured by different sensors on San Francisco Bay area freeways. Exogenous variables are'
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
    'exogenous': [
        'This Exogenous variable is the other sensors, which represent the traffic occupancy data captured by various sensors placed along different roads. The traffic occupancy rates from these sensors are often interrelated, with fluctuations in one sensor’s readings potentially affecting those of others due to shared traffic flow patterns.',
        'Exogenous other sensors capture data from additional sensors along San Francisco Bay area freeways, where their readings may reflect similar traffic congestion trends. These sensors can influence each other’s measurements, particularly during peak traffic times or when road conditions change across the area.',
        'Exogenous other sensors indicate that the traffic occupancy rate measured by one sensor can be impacted by nearby sensors due to common traffic conditions, such as congestion or incidents that affect multiple roads simultaneously.',
        'Exogenous other sensors describe how traffic patterns measured at various points along freeways can be interconnected, with occupancy rates at one sensor often mirroring those at adjacent sensors, as vehicles travel through the network.'
    ],
    'endogenous': [
        'This Endogenous variable is sensor 861, representing the traffic occupancy rate measured at a specific point on the freeway. The occupancy data from this sensor can be influenced by traffic conditions across other nearby sensors, reflecting the flow of vehicles between different parts of the freeway system.',
        'Endogenous sensor 861 captures the traffic occupancy for a particular freeway location, which can be influenced by traffic dynamics observed at other sensors along the route, especially in cases where congestion or incidents affect the entire freeway network.',
        'Endogenous sensor 861 indicates the traffic occupancy rate at a single sensor, which can be correlated with readings from other sensors, as traffic congestion often affects multiple roads or sensors in the area simultaneously.',
        'Endogenous sensor 861 reflects how the traffic occupancy measured at this point on the freeway may mirror or be affected by similar measurements at nearby sensors, showcasing the interdependencies between different points along the San Francisco Bay area freeways.'
    ]
}

prompts = []


for v in exogenous_variables:
    dataset_template1 = dataset_template1+' '+v+','
for nature_attribute in nature_attribute_template['exogenous']:
    prompts.append(nature_attribute)
for trend in trend_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other sensors'+' '+ trend + '.')
for period in period_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other sensors'+' '+ period + '.')
for stability in stability_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other sensors'+' '+ stability + '.')
for noise in noise_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other sensors'+' '+ noise + '.')
for nature_attribute in nature_attribute_template['endogenous']:
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
    print(outputs.last_hidden_state.shape)
    text_emds.append(outputs.last_hidden_state[:,-1])
text_emds = torch.cat(text_emds,dim=0)
torch.save(text_emds, data_name+'.pt')