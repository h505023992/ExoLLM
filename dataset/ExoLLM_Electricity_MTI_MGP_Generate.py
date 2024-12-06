from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer
import torch
import pandas as pd
gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
data_name = 'electricity'
# 定义函数清理非标准字符
def clean_text(text):
    cleaned_text = ''.join([char if ord(char) < 128 else '' for char in text])
    return cleaned_text
# 读取文件
df = pd.read_csv(data_name+'.csv', header=0)#读取的时候读出来了一些非法字符�).', 'This exogenous variable is PAR (�mol/m�/s)

# 打印表头
variables = df.columns[1:].tolist()
for i in range(len(variables )):
    variables[i]  = 'client '+clean_text(variables [i])
exogenous_variables = variables[:-1]
endogenous_variable = 'client 320'
exogenous_variables_text_template = 'Exogenous'
endogenous_template = 'Endogenous'
dataset_template1 = f'This dataset is {data_name}. Values are in kW of each 15 min. Some clients were created after 2011. In these cases consumption were considered zero. All time labels report to Portuguese hour. However all days present 96 measures (24*4). Every year in March time change day (which has only 23 hours) the values between 1:00 am and 2:00 am are zero for all points. Every year in October time change day (which has 25 hours) the values between 1:00 am and 2:00 am aggregate the consumption of two hours. Exogenous variables are'
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
        'This Exogenous variable is the other clients, representing the electricity consumption patterns of individual users. The usage patterns of clients may exhibit similarities or interdependencies due to shared habits, seasons, or external factors.',
        'Exogenous other clients reflects the varying energy demands of users, where multiple clients may follow similar usage trends, leading to potential correlations in their consumption patterns.',
        'Exogenous other clients captures the collective behavior of users whose consumption can be influenced by common external factors, such as weather conditions or price fluctuations, resulting in similar demand patterns across clients.',
        'Exogenous other clients indicates that electricity usage patterns among clients may be interrelated, where fluctuations in one client’s consumption could potentially influence or coincide with others, reflecting shared environmental or social conditions.'
    ],
    'endogenous': [
        'This Endogenous variable is client 320, representing the specific electricity consumption pattern of client 320. The consumption is influenced by both individual factors and the collective consumption behavior of other clients in the network.',
        'Endogenous client 320 reflects the demand for electricity by a particular client, where usage patterns can be influenced by both personal habits and broader usage trends observed in other clients, such as similar time-of-day usage or seasonal demand peaks.',
        'Endogenous client 320 captures the unique consumption dynamics of this individual client, which may be affected by external factors shared with other clients, such as weather or social events that influence electricity consumption.',
        'Endogenous client 320 represents the electricity demand of a single client, whose consumption is interrelated with the broader energy consumption trends in the network, reflecting possible correlations in usage patterns between this client and other users.'
    ]
}
prompts = []


for v in exogenous_variables:
    dataset_template1 = dataset_template1+' '+v+','
for nature_attribute in nature_attribute_template['exogenous']:
    prompts.append(nature_attribute)
for trend in trend_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other clients'+' '+ trend + '.')
for period in period_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other clients'+' '+ period + '.')
for stability in stability_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other clients'+' '+ stability + '.')
for noise in noise_template:
    prompts.append(exogenous_variables_text_template+' '+ 'other clients'+' '+ noise + '.')
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
    text_emds.append(outputs.last_hidden_state[:,-1])
text_emds = torch.cat(text_emds,dim=0)
torch.save(text_emds, data_name+'.pt')