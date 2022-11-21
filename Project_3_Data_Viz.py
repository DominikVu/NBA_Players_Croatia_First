#!/usr/bin/env python
# coding: utf-8

# # NBA Players - Croatia First
# - The analysis was made with the aim of exploring NBA Players (1996-22) database and finding the best Croatian players in selected time period!

# ![dataset-cover.jpg](attachment:dataset-cover.jpg)

# The data set contains over two decades of data on each player who has been part of an NBA teams' roster. It captures demographic variables such as age, height, weight and place of birth, biographical details like the team played for, draft year and round. In addition, it has box score statistics such as games played, average number of points, rebounds, assists, etc.

# Importing the Data we got from: https://www.kaggle.com/datasets/justinas/nba-players-data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns
import os
import gc


# In[2]:


df = pd.read_csv("all_seasons.csv")


# ## Let's start by getting to know the data 

# In[3]:


df


# In[4]:


df.info() 


# In[5]:


df.columns


# In[6]:


df.describe(include="O") 


# To make easier for readers of this analysis to understand the data, here are the descriptions of some column names in the df:
# - pts : Average number of points scored (points_avg)
# - reb : Average number of rebounds grabbed (rebounds_avg)
# - ast : Average number of assists distributed (assists_avg)
# - ts_pct : Measure of the player's shooting efficiency that takes into account free throws, 2 and 3 point shots (shooting_eff_pct)
# - ast_pct : Percentage of teammate field goals the player assisted while he was on the floor (assisted_goals_pct)

# ##  Data Cleaning

# In[7]:


# since we do not need these columns for out analysis
df.drop(['Unnamed: 0',
         'draft_year', 
         'draft_round', 
         'draft_number', 
         'college','oreb_pct', 
         'dreb_pct', 
         'usg_pct','net_rating'], axis=1, inplace=True)


# In[8]:


df.head(3) 


# In[9]:


df.team_abbreviation.unique() 
# Seems like there are to many clubs -> there should be 30 


# In[10]:


df.team_abbreviation.replace([ 'NOK' , 'CHH' , 'NJN' , 'NOH' , 'SEA' , 'VAN'], 
                             value=["NOP", "CHA", "BKN", "NOP", "OKC", "UTA"], inplace=True)


# In[11]:


len(df.team_abbreviation.unique() )


# In[12]:


len(df.player_name.unique())


# In[13]:


# renaming the columns for clearer understanding
df.rename(columns= {"gp":"games_season", 
                    "pts":"points_avg",
                    "reb":"rebounds_avg",
                    "ast":"assists_avg",
                    "ts_pct":"shooting_eff_pct",
                    "ast_pct":"assisted_goals_pct"}, inplace=True)


# In[14]:


# check out the new columns
df.columns


# In[15]:


df.duplicated(keep="first").sum() 
# nice!


# ## Distribution points_avg vs rebounds_avg

# In[32]:


# for all the players in our df 
sns.histplot(df.groupby('player_name').mean()["points_avg"], discrete=True, kde=True) 


# In[33]:


# for all the players in our df 
sns.histplot(df.groupby('player_name').mean()["shooting_eff_pct"], discrete=False, kde=True) 


# In[18]:


df.columns


# ## Correlation 

# In[34]:


corr = df.corr()
matrix = np.triu(corr) 

plt.figure() 
sns.heatmap(round(corr,2), mask=matrix, annot=True) 
plt.show() 
# now when we see that the correlation between player_height and points_avg is close to zero 


# In[20]:


df.sort_values(by="season")


# ## Regplot player_height

# In[35]:


season_96_97 = df.loc[(df['season'] == '1996-97')]
season_21_22 = df.loc[(df['season'] == '2021-22')]
df_grouped = df.groupby('player_name').mean()

# A figure with three regplots
fig1, axs1 = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12,8))
sns.regplot(x='player_height', y='points_avg', 
            fit_reg=True, color='yellow', 
            data=season_96_97, line_kws={"color":"r","alpha":0.7,"lw":5}, 
            ax=axs1[0]).set_title('Points\nHeight, season_96_97')
            
sns.regplot(x='player_height', y='points_avg', 
            fit_reg=True, color='tab:red', 
            data=season_21_22, 
            line_kws={"color":"yellow","alpha":0.7,"lw":5}, 
            ax=axs1[1]).set_title('Points\nHeight, season_21_22')

sns.regplot(x='player_height', y='points_avg', 
            fit_reg=True, color='yellow', 
            data=df_grouped, 
            line_kws={"color":"r","alpha":0.7,"lw":5}, 
            ax=axs1[2]).set_title('Points\nHeight, season_all')
 
# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()
# Interesting! Seems like the height isn't the only factor


# In[36]:


# for these plots, we will use variables with high negative correlation!
fig1, axs1 = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12,8))
sns.regplot(x='player_height', y='assisted_goals_pct', 
            fit_reg=True, color='tab:blue', 
            data=season_96_97, line_kws={"color":"yellow","alpha":0.7,"lw":5}, 
            ax=axs1[0]).set_title('Points\nHeight, season_96_97')
            
sns.regplot(x='player_height', y='assisted_goals_pct', 
            fit_reg=True, color='yellow', 
            data=season_21_22, 
            line_kws={"color":"b","alpha":0.7,"lw":5}, 
            ax=axs1[1]).set_title('Points\nHeight, season_21_22')

sns.regplot(x='player_height', y='assisted_goals_pct', 
            fit_reg=True, color='tab:blue', 
            data=df_grouped, 
            line_kws={"color":"yellow","alpha":0.7,"lw":5}, 
            ax=axs1[2]).set_title('Points\nHeight, season_all')
 
# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()
# Interesting! Seems like the height here is a bit more significant factor, but still a very low one


# ## Lets check all the Croatian players by some metrics to find the best ones!

# In[23]:


# all Croatian players 
df[(df["country"] == "Croatia")].groupby("player_name").mean()


# In[24]:


df[(df["country"] == "Croatia")]["player_name"].unique()


# In[25]:


import plotly.graph_objects as go

# setting the data for y 
ton = df[(df["country"] == "Croatia") & (df["player_name"] == "Toni Kukoc")]['points_avg']
bru = df[(df["country"] == "Croatia") & (df["player_name"] == "Bruno Sundov")]['points_avg']
dal = df[(df["country"] == "Croatia") & (df["player_name"] == "Dalibor Bagaric")]['points_avg']
gor = df[(df["country"] == "Croatia") & (df["player_name"] == "Gordan Giricek")]['points_avg']
zor = df[(df["country"] == "Croatia") & (df["player_name"] == "Zoran Planinic")]['points_avg']
kas = df[(df["country"] == "Croatia") & (df["player_name"] == "Mario Kasun")]['points_avg']
boj = df[(df["country"] == "Croatia") & (df["player_name"] == "Bojan Bogdanovic")]['points_avg']
dam = df[(df["country"] == "Croatia") & (df["player_name"] == "Damjan Rudez")]['points_avg']
hez = df[(df["country"] == "Croatia") & (df["player_name"] == "Mario Hezonja")]['points_avg']
duj = df[(df["country"] == "Croatia") & (df["player_name"] == "Duje Dukan")]['points_avg']
dar = df[(df["country"] == "Croatia") & (df["player_name"] == "Dario Saric")]['points_avg']
dra = df[(df["country"] == "Croatia") & (df["player_name"] == "Dragan Bender")]['points_avg']
ivi = df[(df["country"] == "Croatia") & (df["player_name"] == "Ivica Zubac")]['points_avg']
ant = df[(df["country"] == "Croatia") & (df["player_name"] == "Ante Zizic")]['points_avg']
luk = df[(df["country"] == "Croatia") & (df["player_name"] == "Luka Samanic")]['points_avg']

# names of all the Croatian players
x_data = ['Toni Kukoc', 'Bruno Sundov', 'Dalibor Bagaric', 'Gordan Giricek',
       'Zoran Planinic', 'Mario Kasun', 'Bojan Bogdanovic',
       'Damjan Rudez', 'Mario Hezonja', 'Duje Dukan', 'Dario Saric',
       'Dragan Bender', 'Ivica Zubac', 'Ante Zizic', 'Luka Samanic']

# list for looping
y_data = [ton, bru, dal, gor, zor, kas, boj, 
        dam, hez, duj, dar, dra, ivi, ant, luk]

# selecting the color palettes for our boxplots 
colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)', 
         'rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)']

fig = go.Figure()

# zipping the lists and looping! 
for xd, yd, cls in zip(x_data, y_data, colors):
        fig.add_trace(go.Box(
            name=xd,
            y=yd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker_size=2,
            line_width=1)
        )

fig.update_layout(
        title='Average Points Scored by Croatian NBA Players',
        yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig.show()


# In[26]:


# Let's take top 4 by median and make a  -> Toni, Bojan, Gordan & Dario
# we are going to use the following columns: 
# 'points_avg', 'rebounds_avg', 'assists_avg', 'shooting_eff_pct', 'assisted_goals_pct'


# In[27]:


df.columns 


# In[28]:


from math import pi

# now we need to create a list of lists, containg the key metrics for the best Croatian players (96-22)
points_lst = list(df[(df["country"] == "Croatia") & df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])].groupby("player_name").mean()["points_avg"])
#print(points_lst)

reb_lst = list(df[(df["country"] == "Croatia") & df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])].groupby("player_name").mean()["rebounds_avg"])
#print(reb_lst)

assist_avg_lst = list(df[(df["country"] == "Croatia") & df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])].groupby("player_name").mean()["assists_avg"])
#print(assist_avg_lst)

shoot_lst = list(df[(df["country"] == "Croatia") & df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])].groupby("player_name").mean()["shooting_eff_pct"])
#print(shoot_lst)

assist_goal_lst = list(df[(df["country"] == "Croatia") & df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])].groupby("player_name").mean()["assisted_goals_pct"])
#print(assist_goal_lst)

list_players_data = [points_lst, reb_lst, assist_avg_lst, shoot_lst, assist_goal_lst]
list_players_data 


# ## Some of Best Croatian NBA players in Spider charts!

# In[29]:


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=5):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

# setting the range
ranges = [(5, 16), (2, 6), (1, 4),
         (0.35, 0.8), (0, 0.25)]   

# selecting the variables 
variables = ("points_avg", "rebounds_avg", "assists_avg", 
            "shooting_eff_pct", "assisted_goals_pct") 
players = ['Bojan Bogdanovic\n','Dario Saric\n','Gordan Giricek\n','Toni Kukoc\n']

# creating n (in this case 4) radar charts with data from above
for j in range(len(list_players_data[0])):
    plt.style.use("dark_background")
    data = [i[j] for i in list_players_data]
    fig1 = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig1, variables, ranges)
    radar.plot(data, color="yellow")
    radar.fill(data, alpha=0.2,color="yellow")
    plt.title(players[j], fontsize = 15, color="yellow")   
    plt.show()     
 


# ## Visualization of individual performances through out a career in the NBA

# In[30]:


# data for the creation of lineplots
p1 = pd.DataFrame(df[(df["country"] == "Croatia") & 
   df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])])[["player_name","season","points_avg"]]

p2 = pd.DataFrame(df[(df["country"] == "Croatia") & 
   df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])])[["player_name","season","rebounds_avg"]]

p3 = pd.DataFrame(df[(df["country"] == "Croatia") & 
   df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])])[["player_name","season","assists_avg"]]

p4 = pd.DataFrame(df[(df["country"] == "Croatia") & 
   df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])])[["player_name","season","shooting_eff_pct"]]

p5 = pd.DataFrame(df[(df["country"] == "Croatia") & 
   df["player_name"].isin(["Toni Kukoc", "Bojan Bogdanovic", 
 "Gordan Giricek", "Dario Saric"])])[["player_name","season","assisted_goals_pct"]]


# In[51]:


fig, ax = plt.subplots(5, 1, figsize=(12,16))

# points_avg
ax1 = sns.lineplot(x="season", y="points_avg",
             hue="player_name", data=p1, ax = ax[0])
ax1.set(xticklabels=[])  
ax1.set(title='points_avg: all seasons\n')  # add a title
ax1.set_xlabel='season'

# rebounds_avg
ax2 = sns.lineplot(x="season", y="rebounds_avg",
             hue="player_name", data=p2, ax = ax[1])
ax2.set(xticklabels=[])  
ax2.set(title='rebounds_avg: all seasons\n')  # add a title
ax2.set_xlabel='season'

# assists_avg
ax3 = sns.lineplot(x="season", y="assists_avg",
             hue="player_name", data=p3, ax = ax[2])
ax3.set(xticklabels=[])  
ax3.set(title='assists_avg: all seasons\n')  # add a title
ax3.set_xlabel='season'

# shooting_eff_pct
ax4 = sns.lineplot(x="season", y="shooting_eff_pct",
             hue="player_name", data=p4, ax = ax[3])
ax4.set(xticklabels=[]) 
ax4.set(title='shooting_eff_pct: all seasons\n')  # add a title
ax4.set_xlabel='season'

# assisted_goals_pct
ax5 = sns.lineplot(x="season", y="assisted_goals_pct",
             hue="player_name", data=p5, ax = ax[4])
ax5.set(xticklabels=[])  
ax5.set(title='assisted_goals_pct: all seasons\n')  # add a title
ax5.set_xlabel='season'

plt.tight_layout(pad = 3)
plt.show()


#  I hope you enjoyed it, feel free to leave feedback if you want!

# ![NBA_cro.jpg](attachment:NBA_cro.jpg)
