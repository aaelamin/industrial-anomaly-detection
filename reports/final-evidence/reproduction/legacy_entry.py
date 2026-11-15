"""Execute unchanged legacy scripts with an explicit dataset-access boundary."""
import argparse, json, os, runpy, sys
from pathlib import Path
from suite_common import Suite, PACKAGE, read, sha, write, utc

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--script',required=True)
    p.add_argument('--category',default='bottle');p.add_argument('--normal-only',action='store_true');p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args();ctx=Suite(a.config);ctx.check_sources()
    script=(ctx.project/a.script).resolve()
    assert script.is_relative_to(ctx.project) and a.script in ctx.manifest['legacy_sources']
    assert sha(script)==ctx.manifest['legacy_sources'][a.script]
    sources=ctx.data['datasets']['mvtec'];categories=list(sources) if a.category=='all' else [a.category]
    rows=[r for cat in categories for role in (['train'] if a.normal_only else ['train','calibration','evaluation','masks']) for r in sources[cat][role]]
    canon=lambda p:os.path.normcase(str(Path(p).resolve()))
    allowed={canon(ctx.roots['mvtec']/r['path']):r['sha256'] for r in rows}
    roots=[canon(v) for v in ctx.roots.values()];opened=set();verified=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=canon(os.fsdecode(args[0]))
        if not any(path==r or path.startswith(r+os.sep) for r in roots):return
        if path not in allowed:raise PermissionError('Undeclared dataset input: '+path)
        mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_APPEND|os.O_TRUNC)):
            raise PermissionError('Dataset writes prohibited')
        opened.add(path)
    sys.addaudithook(audit)
    for path,h in allowed.items():assert sha(path)==h,path;verified.add(path)
    sys.path.insert(0,str(ctx.project));os.chdir(ctx.project)
    args=a.args[1:] if a.args[:1]==['--'] else a.args
    sys.argv=[str(script),*args]
    status='completed';failure=None
    try:runpy.run_path(str(script),run_name='__main__')
    except BaseException as e:
        if not isinstance(e,SystemExit) or e.code not in (None,0):status='failed';failure=repr(e)
        raise
    finally:
        write(ctx.workspace/'legacy-access'/f'{os.getpid()}-{a.script.replace("/","_")}.json',
              {'utc':utc(),'script':a.script,'categories':categories,'status':status,'failure':failure,
               'normal_only':a.normal_only,'opened':sorted(opened),'hash_verified':len(verified),
               'allowlist_only':True,'reserved_images_opened':0})
if __name__=='__main__':main()
