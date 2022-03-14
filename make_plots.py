import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)

def set_axis_properties(inpAx):
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # For the minor ticks, use no labels; default NullFormatter.
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    return None

###- CNN with IID####
fig,ax=plt.subplots(figsize=(4,4),ncols=1,nrows=1)
for B in [10,50,600]:
    for E in [1,5]:
        model='cnn'
        exp_res_inpF='./save/exp_{}_B{}_E{}_C0.1_iid1.csv'.format(model,B,E)
        df=pd.read_csv(exp_res_inpF)
        if B==600:
            labelStr='B={} E={} '.format('$\infty$',E)
        else:
            labelStr='B={} E={} '.format(B,E)
        ax.plot(range(200),df['test_acc'],label=labelStr)


ax.axhline(y=0.99,linestyle='dashed',color='red',linewidth=2)
ax.set_ylim([0.8,1])
ax.set_xlabel('Rounds')
ax.set_ylabel('Test Accuracy')
set_axis_properties(ax)
ax.legend(loc='lower right', frameon=False)


for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
plt.tight_layout()
fig.savefig('save/cnn-iid.pdf')
fig.savefig('save/cnn-iid.png',dpi=300)


###- CNN with non-IID####

fig,ax=plt.subplots(figsize=(4,4),ncols=1,nrows=1)
for B in [10,50]:
    for E in [1,5]:
        model='cnn'
        iid_set=0
        exp_res_inpF='./save/exp_{}_B{}_E{}_C0.1_iid0.csv'.format(model,B,E)
        try:
            df=pd.read_csv(exp_res_inpF)
            if B==600:
                labelStr='B={} E={} '.format('$\infty$',E)
            else:
                labelStr='B={} E={} '.format(B,E)
            ax.plot(range(200),df['test_acc'],label=labelStr)
        except:
            continue

ax.axhline(y=0.99,linestyle='dashed',color='red',linewidth=2)
ax.set_ylim([0.8,1])
# ax.set_xlim([0,1])
ax.set_xlabel('Rounds')
ax.set_ylabel('Test Accuracy')
set_axis_properties(ax)
ax.legend(loc='lower right', frameon=False)


for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
plt.tight_layout()
fig.savefig('save/cnn-non-iid.pdf')
fig.savefig('save/cnn-non-iid.png',dpi=300)



fig,ax=plt.subplots(figsize=(4,4),ncols=1,nrows=1)
B=10
E=5


exp_res_inpF='./save/exp_{}_B{}_E{}_C0.1_iid1.csv'.format('cnn',B,E)
df_cnn=pd.read_csv(exp_res_inpF)
exp_res_inpF='./save/exp_{}_B{}_E{}_C0.1_iid1.csv'.format('mlp',B,E)
df_mlp=pd.read_csv(exp_res_inpF)

ax.plot(range(200),df_cnn['test_acc'],label='cnn B=10 E=5')
ax.plot(range(200),df_mlp['test_acc'],label='mlp B=10 E=5')


    # ax.plot(t[:800],rir_file[:800],color='red')
ax.axhline(y=0.99,linestyle='dashed',color='red',linewidth=2)
ax.set_ylim([0.8,1])
# ax.set_xlim([0,1])
ax.set_xlabel('Rounds')
ax.set_ylabel('Test Accuracy')
set_axis_properties(ax)
ax.legend(loc='lower right', frameon=False)


for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
plt.tight_layout()
fig.savefig('save/cnn-mlp.pdf')
fig.savefig('save/cnn-mlp.png',dpi=300)
