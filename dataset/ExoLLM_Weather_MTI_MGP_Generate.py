from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer
import torch
import pandas as pd
gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
data_name = 'weather'

def clean_text(text):
    cleaned_text = ''.join([char if ord(char) < 128 else '' for char in text])
    return cleaned_text

df = pd.read_csv(data_name+'.csv', header=0)
variables = df.columns[1:].tolist()
for i in range(len(variables )):
    variables[i]  = clean_text(variables [i])
exogenous_variables = variables[:-1]
endogenous_variable = 'CO2 (ppm)'
exogenous_variables_text_template = 'Exogenous'
endogenous_template = 'Endogenous'
dataset_template1 = f'This dataset is {data_name} which is recorded every 10 minutes for 2020 whole year, containing 21 meteorological indicators. Exogenous variables are'
dataset_template2 = f' and it is necessary to utilize these exogenous variables sequentially to predict the endogenous variable'
print(exogenous_variables)

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
    'p (mbar)': [
        'This Exogenous variable is p (mbar), representing atmospheric pressure, which affects CO2 (ppm) by influencing gas density and the dispersion of carbon dioxide in the atmosphere.',
        'Exogenous p (mbar) can indirectly impact CO2 (ppm) through pressure-driven air circulation patterns.',
        'Exogenous p (mbar) reflects barometric variations, which play a role in the vertical distribution of CO2 (ppm).',
        'Exogenous p (mbar) might modulate CO2 (ppm) levels by altering the stability of atmospheric layers.'
    ],
    'T (degC)': [
        'This Exogenous variable is T (degC), representing air temperature, which affects CO2 (ppm) by influencing gas solubility and photosynthetic activity.',
        'Exogenous T (degC) directly interacts with CO2 (ppm) by altering plant respiration and carbon exchange rates.',
        'Exogenous T (degC) contributes to the fluctuation of CO2 (ppm) through its effect on biological and chemical processes.',
        'Exogenous T (degC) is a crucial driver of CO2 (ppm) variation, especially under temperature-sensitive atmospheric conditions.'
    ],
    'Tpot (K)': [
        'This Exogenous variable is Tpot (K), denoting potential temperature, which influences CO2 (ppm) by accounting for adiabatic temperature changes affecting carbon dioxide dispersion.',
        'Exogenous Tpot (K) provides insights into the vertical stability of the atmosphere, which can shape CO2 (ppm) profiles.',
        'Exogenous Tpot (K) may indicate how thermal stratification impacts the spatial distribution of CO2 (ppm).',
        'Exogenous Tpot (K) refines the understanding of CO2 (ppm) variations under dynamic atmospheric conditions.'
    ],
    'Tdew (degC)': [
        'This Exogenous variable is Tdew (degC), representing dew point temperature, which impacts CO2 (ppm) by indicating moisture levels that influence carbon cycle dynamics.',
        'Exogenous Tdew (degC) indirectly shapes CO2 (ppm) by affecting stomatal conductance and plant activity.',
        'Exogenous Tdew (degC) serves as a marker of humidity, which modulates CO2 (ppm) through its role in vegetation and microbial processes.',
        'Exogenous Tdew (degC) might suggest conditions that influence the absorption or release of CO2 (ppm) in the environment.'
    ],
    'rh (%)': [
        'This Exogenous variable is rh (%), representing relative humidity, which affects CO2 (ppm) by altering photosynthetic efficiency and stomatal behavior.',
        'Exogenous rh (%) interacts with CO2 (ppm) by regulating water and gas exchange in plants.',
        'Exogenous rh (%) is an indicator of moisture conditions that modulate CO2 (ppm) via biological and chemical pathways.',
        'Exogenous rh (%) reflects atmospheric humidity levels, which can drive variations in CO2 (ppm) concentrations.'
    ],
    'VPmax (mbar)': [
        'This Exogenous variable is VPmax (mbar), denoting the maximum vapor pressure, which influences CO2 (ppm) by indicating potential evaporation and its effects on plant processes.',
        'Exogenous VPmax (mbar) affects CO2 (ppm) by driving water vapor dynamics that intersect with carbon dioxide exchange.',
        'Exogenous VPmax (mbar) provides clues to conditions that regulate CO2 (ppm) through evapotranspiration and climate interactions.',
        'Exogenous VPmax (mbar) plays a role in shaping CO2 (ppm) via its link to atmospheric moisture potential.'
    ],
    'VPact (mbar)': [
        'This Exogenous variable is VPact (mbar), denoting the actual vapor pressure, which impacts CO2 (ppm) by reflecting the ambient moisture content influencing gas exchange processes.',
        'Exogenous VPact (mbar) interacts with CO2 (ppm) by regulating transpiration and atmospheric moisture dynamics.',
        'Exogenous VPact (mbar) signals environmental water vapor levels that affect CO2 (ppm) through plant and microbial processes.',
        'Exogenous VPact (mbar) contributes to CO2 (ppm) variations by indicating real-time atmospheric humidity conditions.'
    ],
    'VPdef (mbar)': [
        'This Exogenous variable is VPdef (mbar), representing vapor pressure deficit, which affects CO2 (ppm) by influencing stomatal closure and transpiration rates.',
        'Exogenous VPdef (mbar) indicates atmospheric dryness, which can modulate CO2 (ppm) through its impact on vegetation and soil respiration.',
        'Exogenous VPdef (mbar) reflects the difference between actual and maximum vapor pressure, shaping CO2 (ppm) through water stress conditions.',
        'Exogenous VPdef (mbar) might drive CO2 (ppm) changes by altering carbon exchange rates in plants.'
    ],
    'sh (g/kg)': [
        'This Exogenous variable is sh (g/kg), representing specific humidity, which affects CO2 (ppm) by indicating the amount of water vapor that can influence gas diffusion.',
        'Exogenous sh (g/kg) contributes to CO2 (ppm) dynamics by reflecting the atmospheric moisture content.',
        'Exogenous sh (g/kg) impacts CO2 (ppm) by modulating photosynthetic and microbial activity through humidity levels.',
        'Exogenous sh (g/kg) shapes CO2 (ppm) by influencing evapotranspiration and plant carbon exchange.'
    ],
    'H2OC (mmol/mol)': [
        'This Exogenous variable is H2OC (mmol/mol), representing water vapor concentration, which impacts CO2 (ppm) by affecting the atmospheric moisture-gas interaction.',
        'Exogenous H2OC (mmol/mol) directly influences CO2 (ppm) by altering the rate of gas exchange and diffusion.',
        'Exogenous H2OC (mmol/mol) provides critical insights into CO2 (ppm) dynamics under varying water vapor concentrations.',
        'Exogenous H2OC (mmol/mol) modulates CO2 (ppm) through its role in photosynthesis and transpiration processes.'
    ],
    'rho (g/m**3)': [
        'This Exogenous variable is rho (g/m**3), representing air density, which affects CO2 (ppm) by influencing the dispersion and transport of carbon dioxide in the atmosphere.',
        'Exogenous rho (g/m**3) shapes CO2 (ppm) variations through its role in atmospheric mixing and gas distribution.',
        'Exogenous rho (g/m**3) provides a measure of air mass concentration, which can affect CO2 (ppm) indirectly.',
        'Exogenous rho (g/m**3) impacts CO2 (ppm) by modulating the vertical and horizontal flow of gases.'
    ],
    'wv (m/s)': [
        'This Exogenous variable is wv (m/s), representing wind velocity, which affects CO2 (ppm) by influencing the horizontal and vertical transport of carbon dioxide.',
        'Exogenous wv (m/s) contributes to CO2 (ppm) dynamics by driving the mixing of atmospheric layers.',
        'Exogenous wv (m/s) provides insights into CO2 (ppm) variations under different wind conditions.',
        'Exogenous wv (m/s) modulates CO2 (ppm) through its role in diffusion and dispersion.'
    ],
    'max. wv (m/s)': [
        'This Exogenous variable is max. wv (m/s), representing maximum wind velocity, which impacts CO2 (ppm) by indicating peak atmospheric mixing potential.',
        'Exogenous max. wv (m/s) can drive significant variations in CO2 (ppm) through extreme wind events.',
        'Exogenous max. wv (m/s) reflects atmospheric dynamics that influence the spatial distribution of CO2 (ppm).',
        'Exogenous max. wv (m/s) might shape CO2 (ppm) patterns through its effect on turbulent flow and transport.'
    ],
    'wd (deg)': [
        'This Exogenous variable is wd (deg), representing wind direction, which affects CO2 (ppm) by indicating the source and pathway of air masses.',
        'Exogenous wd (deg) contributes to CO2 (ppm) dynamics by shaping the transport routes of carbon dioxide.',
        'Exogenous wd (deg) provides insights into regional and local variations in CO2 (ppm) concentrations.',
        'Exogenous wd (deg) modulates CO2 (ppm) through its role in atmospheric circulation patterns.'
    ],
    'rain (mm)': [
        'This Exogenous variable is rain (mm), representing precipitation, which affects CO2 (ppm) by influencing soil carbon release and atmospheric washout.',
        'Exogenous rain (mm) contributes to CO2 (ppm) dynamics by altering microbial activity and carbon fluxes.',
        'Exogenous rain (mm) reflects hydrological conditions that shape CO2 (ppm) through deposition and removal processes.',
        'Exogenous rain (mm) modulates CO2 (ppm) by interacting with vegetation and soil respiration rates.'
    ],
    'raining (s)': [
        'This Exogenous variable is raining (s), representing the duration of precipitation, which impacts CO2 (ppm) by indicating sustained moisture conditions.',
        'Exogenous raining (s) contributes to CO2 (ppm) dynamics through its prolonged influence on soil and vegetation processes.',
        'Exogenous raining (s) reflects extended hydrological effects that modulate CO2 (ppm) through carbon cycle interactions.',
        'Exogenous raining (s) impacts CO2 (ppm) by driving temporal variations in precipitation-related factors.'
    ],
'SWDR (W/m)': [
    'This Exogenous variable is SWDR (W/m), representing shortwave downward radiation, which impacts CO2 (ppm) by influencing photosynthetic activity and energy fluxes.',
    'Exogenous SWDR (W/m) drives CO2 (ppm) variation through its role in determining solar energy available for plant growth.',
    'Exogenous SWDR (W/m) reflects the intensity of sunlight, which can modulate CO2 (ppm) via photosynthesis and surface energy exchanges.',
    'Exogenous SWDR (W/m) contributes to the dynamics of CO2 (ppm) by shaping radiative energy input into the ecosystem.'
],
'PAR (mol/m/s)': [
    'This Exogenous variable is PAR (mol/m/s), representing photosynthetically active radiation, which affects CO2 (ppm) by providing energy for photosynthesis.',
    'Exogenous PAR (mol/m/s) directly interacts with CO2 (ppm) by driving plant carbon fixation rates.',
    'Exogenous PAR (mol/m/s) modulates CO2 (ppm) through its influence on the efficiency of light-dependent reactions in photosynthesis.',
    'Exogenous PAR (mol/m/s) shapes CO2 (ppm) dynamics by determining the availability of light for carbon assimilation in plants.'
],
'max. PAR (mol/m/s)': [
    'This Exogenous variable is max. PAR (mol/m/s), representing the maximum photosynthetically active radiation, which impacts CO2 (ppm) by reflecting peak light conditions for photosynthesis.',
    'Exogenous max. PAR (mol/m/s) drives CO2 (ppm) variation by indicating the maximum potential for carbon fixation during high-light periods.',
    'Exogenous max. PAR (mol/m/s) reflects peak radiative input that can modulate CO2 (ppm) through enhanced photosynthetic activity.',
    'Exogenous max. PAR (mol/m/s) contributes to the dynamics of CO2 (ppm) by shaping the upper limit of light-driven carbon assimilation.'
],
'Tlog (degC)': [
    'This Exogenous variable is Tlog (degC), representing the logarithmic transformation of air temperature, which affects CO2 (ppm) by emphasizing temperature variations relevant to biological and chemical processes.',
    'Exogenous Tlog (degC) provides a refined perspective on temperature-driven CO2 (ppm) dynamics by highlighting non-linear thermal effects.',
    'Exogenous Tlog (degC) modulates CO2 (ppm) by influencing temperature-sensitive processes such as respiration, photosynthesis, and soil carbon release.',
    'Exogenous Tlog (degC) shapes CO2 (ppm) variations by capturing critical thermal thresholds that impact carbon exchange rates.'
],
'CO2 (ppm)': [
    'This Endogenous variable is CO2 (ppm), representing the concentration of carbon dioxide, a key indicator of atmospheric carbon cycles and ecosystem dynamics.',
    'Endogenous CO2 (ppm) reflects the balance between carbon emissions and uptake, directly influenced by both natural and anthropogenic factors.',
    'Endogenous CO2 (ppm) serves as a critical measure of the system’s carbon budget, with variations signaling changes in photosynthesis, respiration, and combustion processes.',
    'Endogenous CO2 (ppm) encapsulates the dynamic interactions of biological, physical, and chemical processes that regulate atmospheric carbon content.'
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