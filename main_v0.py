#%%
import pandas as pd
import sys
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict
#%%
cwd = os.path.dirname(os.path.realpath(__file__))
vacdat_file_path = os.path.join(cwd,'covid_vaccination_progress','country_vaccinations.csv')

vacdat = pd.read_csv(vacdat_file_path)
total_records = vacdat.shape[0]
vacdat.head()
#%%
countries = list(set(vacdat['country']))
countries.sort()
print(countries)
# %%
nan_remove = ['total_vaccinations',
            'people_vaccinated',
            'people_fully_vaccinated',
            'daily_vaccinations_raw',
            'daily_vaccinations',
            'people_vaccinated_per_hundred',
            'people_fully_vaccinated_per_hundred',
            'daily_vaccinations_per_million']
for col_name in nan_remove:
    vacdat[col_name].fillna(0, inplace=True)
vacdat['date'] = pd.to_datetime(vacdat['date'])
vacdat.head()
# %%
print(set(vacdat['vaccines'].values))
# %%
vac_tags = ['Oxford',
            'Pfizer',
            'Sinopharm-W',
            'Sinopharm-B',
            'Sinovac',
            'Sputnik V',
            'Moderna',
            'Covaxin']
vac_hash = OrderedDict()
vac_hash['Oxford/AstraZeneca'] = 'Oxford'
vac_hash['Pfizer/BioNTech'] = 'Pfizer'
vac_hash['Sinopharm/Wuhan'] = 'Sinopharm-W'
vac_hash['Sinopharm/Beijing'] = 'Sinopharm-B'
vac_hash['Sinovac'] = 'Sinovac'
vac_hash['Sputnik V'] = 'Sputnik V'
vac_hash['Moderna'] = 'Moderna'
vac_hash['Covaxin'] = 'Covaxin'
# %%
vac_series = {}
for tag in vac_tags:
    vac_series[tag] = [0]*total_records
# %%
new_vac_col = []
row_indx = 0
for record in vacdat['vaccines'].values:
    components = record.replace(', ',',').split(",")
    new_components = []
    for sub_component in components:
        if sub_component in vac_hash.keys():
            new_components.append(vac_hash[sub_component])
            vac_series[vac_hash[sub_component]][row_indx] = 1
        else:
            msg = "Warning. vacccine name {} not found in hash".format(sub_component)
            print(msg)
            break
        new_vac_col.append(new_components)
    row_indx+=1
        


# %%
insert_loc = 13
for j, vac_tag in enumerate(vac_tags):
    vacdat.insert(insert_loc + j, vac_tag, vac_series[vac_tag])
vacdat.head()


# %%
vac_tag_in_country = {}
for vac_tag in vac_tags:
    vac_tag_in_country[vac_tag] = []
# %%
# %%
power_countries = [
    'China',
    'United States',
    'Germany',
    'India',
    'Italy',
    'France',
    'United Kingdom',
    'Brazil',
    'Canada',
    'Russia',
    'Israel']
power_countries_population = OrderedDict()    
power_countries_population['China'] = 1.398E9
power_countries_population['United States'] = 328.2E6
power_countries_population['Germany'] = 83.02E6
power_countries_population['India'] = 1.366E9
power_countries_population['Italy'] = 60.36E6
power_countries_population['France'] = 67.06E6
power_countries_population['United Kingdom'] = 66.65E6
power_countries_population['Brazil'] = 211E6
power_countries_population['Canada'] = 37.59E6
power_countries_population['Russia'] = 144.4E6
power_countries_population['Israel'] = 9.053E6
# %%
total_vaccinations_per_country = OrderedDict()
country_dic = OrderedDict()

country_dic['country'] = []
country_dic['start_date'] = []
country_dic['end_date'] = []
country_dic['avg_vac_speed'] = []
country_dic['total_vacs'] = []
country_dic['reported_days'] = []

country_daily_vacs = {'country': [], 'country_daily_vacs': []}
def delta_days(end_date, start_date):
    time_frame = end_date - start_date
    time_frame = time_frame.astype('timedelta64[D]')
    return time_frame / np.timedelta64(1, 'D')
power_country_indx = []
j = 0    
for country in countries:
    if country in power_countries:
        power_country_indx.append(j)
    sub_frame = vacdat[vacdat['country']==country]
    country_dic['country'].append(country)
    country_dic['total_vacs'].append(sub_frame['total_vaccinations'].values[-1])
    country_dic['start_date'].append(sub_frame['date'].values[0])
    country_dic['end_date'].append(sub_frame['date'].values[-1])
    country_dic['reported_days'].append(delta_days(sub_frame['date'].values[-1], sub_frame['date'].values[0]))
    country_dic['avg_vac_speed'].append(country_dic['total_vacs'][-1]/country_dic['reported_days'][-1])
    country_daily_vacs['country'].append(country)
    country_daily_vacs['country_daily_vacs'].append(sub_frame['daily_vaccinations'].values[:])
    for vac_tag in vac_tags:
        if any(sub_frame[vac_tag].values):
            vac_tag_in_country[vac_tag].append(country)
    j+=1
# %%
country_df = pd.DataFrame(country_dic)
# %%
for tag in vac_tags:
    print("The vaccine '{0}' is being used in {1} countries".format(tag, len(vac_tag_in_country[tag])))

# %%
power_country_df = country_df.iloc[power_country_indx, :]
print(power_country_df)
power_country_df.describe()
# %%
ordered_population = [int(power_countries_population[country]) for country in power_country_df['country'].values]

power_country_df['population'] = ordered_population
power_country_df['normalized_vac_speed'] = power_country_df['avg_vac_speed']/power_country_df['population']

print(power_country_df)
# %%
horizontal_grid_steps = 12
class BarMaker:
    def __init__(self, country_list, title, x_step=12, width=0.35):
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches((6, 4))
        self.ax.xaxis.grid(True, linestyle='--', which='major',
                        color='grey', alpha=.8)
        self.horizontal_grid_steps = x_step
        self.fig.tight_layout()
        self.fig.set_dpi(200)
        self.ax.invert_yaxis()
        self.country_list = country_list
        self.ypos = np.arange(len(country_list))
        self.width = width
        self.ax.set_title(title)

    def set_y(self, bar_values, legend=[None]):
        if len(bar_values) == 1:
            self.ax.barh(self.ypos , bar_values[0], align='center', label=legend[0])
        else:
            self.ax.barh(self.ypos - self.width/2, bar_values[0], self.width, align='center', label=legend[0])
            self.ax.barh(self.ypos + self.width/2, bar_values[1], self.width, align='center', label=legend[1])            
        self.ax.set_yticks(self.ypos)
        self.ax.set_yticklabels(self.country_list)

    def set_x(self, max_x, str_normalizer, label):
        xaxis_ticks_num = np.linspace(0, max_x, self.horizontal_grid_steps)
        xaxis_ticks_str = [ "{:.1e}".format(number*str_normalizer) for number in xaxis_ticks_num]
        self.ax.set_xticks(xaxis_ticks_num)
        self.ax.set_xticklabels(xaxis_ticks_str, fontsize=6)
        self.ax.set_xlabel(label)
        self.ax.legend()

    def show(self):
        plt.show()
        
power_country_df = power_country_df.sort_values(by='avg_vac_speed', ascending=False)
bar_maker = BarMaker(power_country_df['country'].values, 'Vaccionation Rate By Country')
bar_maker.set_y([power_country_df['avg_vac_speed'].values], legend=['avg_vac_speed'])
bar_maker.set_x(power_country_df['avg_vac_speed'].max(), 1, 'Vaccines Per Day')
bar_maker.show()

# %%
power_country_df = power_country_df.sort_values(by='normalized_vac_speed', ascending=False)
bar_maker = BarMaker(power_country_df['country'].values, 'Vaccionation Rate By Country')
bar_maker.set_y([power_country_df['normalized_vac_speed'].values], legend=['normalized_vac_speed'])
bar_maker.set_x(power_country_df['normalized_vac_speed'].max(), 1, 'Vaccines Per Day Normalized By Population')
# %%
# fig, ax = plt.subplots()
# ypos = range(0, power_country_df.shape[0])
# xaxis_ticks_num = np.linspace(0, power_country_df['normalized_vac_speed'].max(), horizontal_grid_steps)
# xaxis_ticks_str = [ "{:.3f}".format(number*100) for number in xaxis_ticks_num]
# fig.set_dpi(200)
# ax.barh(ypos, power_country_df['normalized_vac_speed'].values, align='center')
# ax.set_yticks(ypos)
# ax.set_yticklabels(power_country_df['country'].values)
# ax.set_xticks(xaxis_ticks_num)
# ax.set_xticklabels(xaxis_ticks_str, fontsize=6)
# ax.invert_yaxis()
# ax.set_xlabel(" Vaccinations per Day Normalized by Population")
# ax.xaxis.grid(True, linestyle='--', which='major',
#                    color='grey', alpha=.8)
# ax.set_title("Geographic Distribution of Vaccinations Per Day")
# fig.tight_layout()

# plt.show()
# %%
# Now do the same graphics but for infections per day normalized by population.
# %%
infdat_file_path = os.path.join(cwd,'covid_daily_confirm','worldometer_coronavirus_daily_data.csv')

infdat = pd.read_csv(infdat_file_path)
inf_total_records = vacdat.shape[0]
infdat.head()
# %%
infdat['country'] = infdat['country'].replace(['UK', 'USA'], ['United Kingdom','United States'])

# %%

infdat['date'] = pd.to_datetime(infdat['date'])
infdat.head()
country_inf_dic =     {'country': [],
    'start_date': [],
    'end_date': [],
    'avg_inf_speed': [],
    'total_inf': [],
    'reported_days': [],
    }
country_daily_inf = {
    'country': [],
    'daily_new_cases': [],
    'active_cases': []}    
power_country_indx = []
j = 0
countries = list(set(infdat['country'].values))
countries.sort()

for country in countries:
    if country in power_countries:
        power_country_indx.append(j)
        sub_frame = infdat[infdat['country']==country]
        sub_frame = sub_frame[sub_frame['date'] > power_country_df['start_date'].min()]
        country_inf_dic['country'].append(country)
        country_inf_dic['total_inf'].append(sub_frame['cumulative_total_cases'].values[-1])
        country_inf_dic['start_date'].append(sub_frame['date'].values[0])
        country_inf_dic['end_date'].append(sub_frame['date'].values[-1])
        country_inf_dic['reported_days'].append(delta_days(sub_frame['date'].values[-1], sub_frame['date'].values[0]))
        country_inf_dic['avg_inf_speed'].append(country_inf_dic['total_inf'][-1]/country_inf_dic['reported_days'][-1])
        country_daily_inf['country'].append(country)
        country_daily_inf['daily_new_cases'].append(sub_frame['daily_new_cases'].values[:])
        country_daily_inf['active_cases'].append(sub_frame['active_cases'].values[:])

        j+=1
# %%
country_inf_df = pd.DataFrame(country_inf_dic)
ordered_population = [int(power_countries_population[country]) for country in country_inf_df['country'].values]
country_inf_df['population'] = ordered_population
# %%
country_inf_df = country_inf_df.sort_values(by='avg_inf_speed', ascending=True)
bar_maker = BarMaker(country_inf_df['country'].values, 'Infection Speed By Country')
bar_maker.set_y([country_inf_df['avg_inf_speed'].values])
bar_maker.set_x(country_inf_df['avg_inf_speed'].max(), 1, 'Infections Per Day')
bar_maker.show()
# %%
country_inf_df['normalized_inf_speed'] = country_inf_df['avg_inf_speed']/country_inf_df['population']


# %%
# plotting graph of normalized infection speed normalized by population
country_inf_df = country_inf_df.sort_values(by='normalized_inf_speed', ascending=True)
bar_maker = BarMaker(country_inf_df['country'].values, 'Infection Speed by Country')
bar_maker.set_y([country_inf_df['normalized_inf_speed'].values])
bar_maker.set_x(country_inf_df['normalized_inf_speed'].max(), 1, 'Normalized Infections Per Day')
bar_maker.show()
# %%
country_inf_df = country_inf_df.sort_values(by='country', ascending=False)
power_country_df = power_country_df.sort_values(by='country', ascending=False)

vacs_inf = OrderedDict()
vacs_inf['country'] = []
vacs_inf['normalized_inf_speed'] = []
vacs_inf['normalized_vac_speed'] = []
for country in power_country_df['country'].values:
    vacs_inf['country'].append(country)
    nis = country_inf_df.loc[country_inf_df['country']==country, 'normalized_inf_speed']
    vacs_inf['normalized_inf_speed'].append(nis.values[0])
    nvs = power_country_df.loc[power_country_df['country']==country, 'normalized_vac_speed']
    vacs_inf['normalized_vac_speed'].append(nvs.values[0])
vacs_inf_df = pd.DataFrame(vacs_inf)
# %%
# fig, ax = plt.subplots()
# ypos = range(0, country_inf_df.shape[0])
# step = country_inf_df['avg_inf_speed'].max()/horizontal_grid_steps
# xaxis_ticks_num = np.linspace(0, country_inf_df['normalized_inf_speed'].max(), horizontal_grid_steps)
# xaxis_ticks_str = [ "{:.3f}".format(number*100) for number in xaxis_ticks_num]
# fig.set_dpi(200)
# ax.barh(ypos, country_inf_df['normalized_inf_speed'].values, align='center')
# ax.set_yticks(ypos)
# ax.set_yticklabels(country_inf_df['country'].values)
# ax.set_xticks(xaxis_ticks_num)
# ax.set_xticklabels(xaxis_ticks_str, fontsize=6)
# ax.invert_yaxis()
# ax.set_xlabel(" Infections / Day Normalized by Population")
# ax.set_title("Geographic Distribution of Infection Rate")
# ax.xaxis.grid(True, linestyle='--', which='major',
#                    color='grey', alpha=.8)
# fig.tight_layout()
# plt.show()
# %%
# fig, ax = plt.subplots()
# ypos = np.arange(country_inf_df.shape[0])

# max_tick = country_inf_df['normalized_inf_speed'].max()
# if power_country_df['normalized_vac_speed'].max() > max_tick:
#     max_tick = power_country_df['normalized_vac_speed'].max()

# xaxis_ticks_num = np.linspace(0, max_tick, horizontal_grid_steps)
# xaxis_ticks_str = [ "{:.3f}".format(number*100) for number in xaxis_ticks_num]
# fig.set_dpi(200)
# width = 0.35
# ax.barh(ypos - width/2, vacs_inf['normalized_inf_speed'], width, align='center', label='inf_speed')
# ax.barh(ypos + width/2, vacs_inf['normalized_vac_speed'], width, align='center', label='vac_speed')
# ax.set_yticks(ypos)
# ax.set_yticklabels(vacs_inf['country'])
# ax.set_xticks(xaxis_ticks_num)
# ax.set_xticklabels(xaxis_ticks_str, fontsize=6)
# ax.invert_yaxis()
# ax.set_xlabel(" Infections / Day Normalized by Population")
# ax.set_title("Geographic Distribution of Infection Rate")
# ax.xaxis.grid(True, linestyle='--', which='major',
#                    color='grey', alpha=.8)
# ax.legend()              
# fig.tight_layout()
# plt.show()
vacs_inf_df = vacs_inf_df.sort_values(by='normalized_vac_speed', ascending=False)
max_tick = vacs_inf_df['normalized_inf_speed'].max()
if vacs_inf_df['normalized_vac_speed'].max() > max_tick:
    max_tick = vacs_inf_df['normalized_vac_speed'].max()

bars = [vacs_inf_df['normalized_inf_speed'].values, vacs_inf_df['normalized_vac_speed'].values]
legends = ['normalized_inf_speed', 'normalized_vac_speed']
    
bar_maker = BarMaker(vacs_inf_df['country'].values, 'Vaccination Rate / Infection Rate Per Country')
bar_maker.set_y(bars, legend=legends)
bar_maker.set_x(max_tick, 1, 'Infections Vs Vaccinations Per Day')
bar_maker.show()
# %%
