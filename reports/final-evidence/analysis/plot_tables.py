"""Publication-size figures from complete, verified saved tables only."""
import argparse,csv,ctypes,hashlib,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;T=D/'team-tables';O=D/'team-plots'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def available_ram():
    if os.name=='nt':
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
        m=Memory();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return m.available
    return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
def rows(name):
    with (T/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    check=read(T/'table-checks.json');assert check['status']=='PASS_COMPLETE_TABLE_EXPORT'
    for n,h in check['files'].items():assert sha(T/n)==h
    assert available_ram()>=3*2**30;start=time.monotonic()
    assert not (O/'figure-checks.json').exists(),'Preserve existing figures; select a new output folder'
    O.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(O/'runtime-cache'))
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    O.mkdir(exist_ok=True);data=rows('main-results.csv');seeds=rows('seed-macro-results.csv')
    metrics=[('image_auroc','Image AUROC'),('pixel_auroc','Pixel AUROC'),('pixel_average_precision','Pixel AP')]
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True,layout='constrained');y=np.arange(len(data));numbers=[]
    for ax,(key,title) in zip(axes,metrics):
        means=np.array([float(r[key+'_mean_percent']) for r in data]);sds=np.array([float(r[key+'_sample_sd_percent']) for r in data])
        ax.errorbar(means,y,xerr=sds,fmt='o',color='#196684',ecolor='#196684',capsize=3,markersize=5,label='Mean ± sample SD')
        for i,row in enumerate(data):
            values=[float(r[key+'_percent']) for r in seeds if r['model']==row['model']];assert len(values)==3
            assert np.isclose(np.mean(values),means[i]) and np.isclose(np.std(values,ddof=1),sds[i])
            ax.scatter(values,np.array([i-.10,i,i+.10]),s=16,color='#b95421',marker='|',zorder=3)
            numbers.append({'model':row['model'],'metric':key,'mean_percent':means[i],'sample_sd_percent':sds[i],'seed_percent':values})
        ax.set_title(title);ax.set_xlim(0,100);ax.set_xlabel('Category-macro score (%)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,[r['label'] for r in data]);axes[0].invert_yaxis()
    fig.suptitle('Fixed-endpoint MVTec comparison: 15 categories × 3 seeds\nBlue: mean ± seed SD; orange marks: individual seed macros',fontsize=12)
    fig.savefig(O/'main-comparison.png',dpi=220);fig.savefig(O/'main-comparison.svg');plt.close(fig)
    delta=rows('paired-category-changes.csv');comparisons=['fusion minus middle','resnet minus week3','week4 minus resnet'];cats=sorted({r['category'] for r in delta})
    matrix=np.array([[float(next(r for r in delta if r['category']==c and r['comparison']==comp and r['metric']=='pixel_average_precision')['mean_change_pp']) for comp in comparisons] for c in cats])
    bound=max(1,float(np.max(np.abs(matrix))));fig,ax=plt.subplots(figsize=(7,7.3),layout='constrained')
    im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(3),['Fusion − middle','Matched RN − original RN','WRN − matched RN'],rotation=15,ha='right');ax.set_yticks(range(15),cats)
    for i in range(15):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.65*bound else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Mean pixel AP change (percentage points)')
    ax.set_title('Category effects of the planned comparisons\nThree-seed means; descriptive, no significance claim',fontsize=12)
    fig.savefig(O/'category-changes.png',dpi=220);fig.savefig(O/'category-changes.svg');plt.close(fig)
    assert time.monotonic()-start<300 and 'torch' not in sys.modules
    write(O/'figure-checks.json',{'status':'PASS_TABLE_TO_FIGURE_VALUES','source_table_check_sha256':sha(T/'table-checks.json'),'main_plot_values':numbers,'category_order':cats,'contrast_order':comparisons,'category_change_matrix_pp':matrix.tolist(),'color_range_pp':[-bound,bound],'files':{p.name:sha(p) for p in O.iterdir() if p.suffix in ['.png','.svg']},'source_images_opened':0,'reserved_images_opened':0,'new_predictions':0,'visual_review_pending':True})
    (O/'CAPTIONS.md').write_text('# Figure captions and interpretation\n\n`main-comparison`: Category-macro image AUROC, pixel AUROC and pixel AP at the fixed endpoint. Each model covers all 15 MVTec categories and three seeds. Error bars are sample SD across seed-level category means; orange marks show those three values. They are not population confidence intervals. All score axes span 0–100%.\n\n`category-changes`: Mean pixel-AP differences across the three seeds for the three planned configuration contrasts. Color is centered at zero with a symmetric range; numeric cells give percentage-point changes. Category-wise patterns are descriptive and not corrected significance tests. The compact fusion pair is matched in flow capacity and representation size. The backbone/recipe contrasts retain the limitations explained in the methods notes.\n\nBoth figures use saved verified metrics only. SVG versions contain editable text; PNG versions are supplied for the report and presentation.\n',encoding='utf-8')
    print('PASS_TABLE_TO_FIGURE_VALUES')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tables',type=Path,default=T);parser.add_argument('--output',type=Path,default=O)
    args=parser.parse_args();T=args.tables.resolve();O=args.output.resolve();main()
