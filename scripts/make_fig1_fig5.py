"""Draw Fig. 1 (workflow) and Fig. 5 (sensitivity of the tuning result) from results/.

    python scripts/make_fig1_fig5.py
"""
import pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
RW=1.37
# ---------- Fig 5: E4 / E5 dot-and-range ----------
import pathlib; RES = pathlib.Path(__file__).resolve().parents[1] / 'results'; FIG = RES.parent / 'figures'
e4=pd.read_csv(RES/'TableIV_E4_fixed_configurations.csv'); e5=pd.read_csv(RES/'TableIV_E5_tightened_space.csv')
lab={'LSTM-PSO (arsip)':'LSTM-PSO\n(Table I)','LSTM-GA (arsip)':'LSTM-GA\n(Table I)',
     'LSTM (standard)':'LSTM\n(default)','LSTM (interior)':'LSTM\n(interior)',
     'PSO-tight':'LSTM-PSO\n(tight)','GA-tight':'LSTM-GA\n(tight)'}
fig,axs=plt.subplots(2,1,figsize=(3.45,5.0))
for ax,df,ttl in ((axs[0],e4,'(a) Fixed configurations'),
                  (axs[1],e5,'(b) Searches re-run in a tightened space')):
    x=range(len(df))
    ax.axhline(RW,color='black',lw=1.2,zorder=1)
    ax.text(-0.4,RW-0.1,'Random walk (1.37)',ha='left',va='top',fontsize=7.5)
    ax.vlines(x,df.MAPE_min,df.MAPE_max,color='0.45',lw=1.6,zorder=2)
    ax.scatter(x,df.MAPE_mean,s=46,color='black',zorder=3,edgecolor='white',linewidth=1.2)
    for xi,(m,s) in enumerate(zip(df.MAPE_mean,df.MAPE_sd)):
        ax.text(xi+0.1,m,f'{m+1e-9:.2f}\n(SD {s+1e-9:.2f})',va='center',fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels([lab[c] for c in df.config],fontsize=7.5)
    ax.set_xlim(-0.5,len(df)-0.15); ax.set_title(ttl,fontsize=9,loc='left')
    ax.grid(axis='y',color='0.9',lw=0.8); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
[(a.set_ylabel('Test MAPE (%)'),a.set_ylim(0,6)) for a in axs]
fig.tight_layout(); fig.savefig(FIG/'Fig5_sensitivity.png',dpi=600)

# ---------- Fig 1: redrawn workflow with larger, high-contrast text ----------
fig,ax=plt.subplots(figsize=(7.0,3.2)); ax.set_xlim(0,100); ax.set_ylim(0,46); ax.axis('off')
W,H=23.6,14
boxes=[('1. Data','SIDO + 5 IDX stocks\n2019–2025; OHLCV,\nJCI, USD/IDR'),
       ('2. Causal features','9 lagged indicators\n4 nested feature sets'),
       ('3. Temporal split','80/20 chronological;\nlast 20% of training\n= validation'),
       ('4. GA / PSO search','7 hyperparameters,\n300 evaluations\n(LSTM only)'),
       ('5. Repeated runs','30 runs per model\nmean ± 95% CI'),
       ('6. Baselines','random walk,\nrandom walk + drift,\nARIMA(1,1,1)'),
       ('7. Significance tests','Diebold–Mariano\n+ Holm correction\nhorizons 1–20'),
       ('8. Visual diagnostics','error bands, stability,\nsignificance vs\nhorizon, cross-stock')]
xs=[0.4,25.6,50.8,76.0]; pos=[(xs[i],29) for i in range(4)]+[(xs[3-i],10) for i in range(4)]
for (t,s),(x,y) in zip(boxes,pos):
    ax.add_patch(FancyBboxPatch((x,y),W,H,boxstyle='round,pad=0,rounding_size=1.2',fc='#E8EEF5',ec='#1F3A5F',lw=1.4))
    ax.text(x+W/2,y+H-2.6,t,ha='center',va='center',fontsize=7.8,fontweight='bold',color='black')
    ax.text(x+W/2,y+H/2-1.8,s,ha='center',va='center',fontsize=7.6,color='black',linespacing=1.15)
ar=dict(arrowstyle='-|>',color='#1F3A5F',lw=1.4,mutation_scale=11)
for i in range(3): ax.annotate('',xy=(xs[i+1],36),xytext=(xs[i]+W,36),arrowprops=ar)
ax.annotate('',xy=(xs[3]+W/2,10+H),xytext=(xs[3]+W/2,29),arrowprops=ar)
for i in range(3): ax.annotate('',xy=(xs[3-i-1]+W,17),xytext=(xs[3-i],17),arrowprops=ar)
ax.add_patch(FancyBboxPatch((14,0.6),72,6.4,boxstyle='round,pad=0,rounding_size=1.2',fc='#FCEBC8',ec='#8A5A00',lw=1.4))
ax.text(50,3.8,'Decision: does any tuned model significantly beat the random walk?',ha='center',va='center',fontsize=8.4,fontweight='bold',style='italic',color='black')
ax.annotate('',xy=(13.8,4.2),xytext=(xs[0]+W/2,10),arrowprops=dict(arrowstyle='-|>',color='#1F3A5F',lw=1.4,mutation_scale=11,connectionstyle='arc3,rad=0.3'))
fig.savefig(FIG/'Fig1_workflow.png',dpi=600,bbox_inches='tight',pad_inches=0.03)
