"""Single3case×3repeat visual follow-up; delegates the frozen native reset audit.

Preserves v1 and the native environment. Captures its actual T=1 image arrays,
visual model state and post-step segmentation for descriptive localization.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,inspect,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

B=Path('/efs/prescottsun/SPT-JEPA')
PARENT=B/'runs/cube_reset_state_audit_20261031/source-v1/experiments/action_projection_20261031/reset_audit.py'
PARENT_SHA='2aac1e7d0ab53f632206809351c1f1ff70b8c5c3f305634576e1410c5ab089f7'
CASES=[3,28,35]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def array_sha(x):
    x=np.ascontiguousarray(x);return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()

def pixel_metrics(a,b,mask=None):
    assert a.shape==b.shape and a.dtype==b.dtype==np.uint8 and a.ndim==3 and a.shape[-1]==3
    delta=a.astype(np.float64)-b.astype(np.float64);d=delta[mask]if mask is not None else delta.reshape(-1,3)
    if not len(d):return dict(pixels=0,mae=None,rmse=None,max_abs=None,changed_pixel_fraction=None,changed_coordinate_fraction=None,over1_pixel_fraction=None,over8_pixel_fraction=None,squared_error_sum=0.)
    ab=np.abs(d)
    return dict(pixels=len(d),mae=float(ab.mean()),rmse=float(np.sqrt(np.square(d).mean())),max_abs=float(ab.max()),
                changed_pixel_fraction=float(np.any(d!=0,axis=-1).mean()),changed_coordinate_fraction=float((d!=0).mean()),
                over1_pixel_fraction=float(np.any(ab>1,axis=-1).mean()),over8_pixel_fraction=float(np.any(ab>8,axis=-1).mean()),squared_error_sum=float(np.square(d).sum()))

def frame_key(index):
    repeat,offset=divmod(index,9)
    if offset<6:return repeat,CASES[offset//2],('initial'if offset%2==0 else'goal')
    return repeat,CASES[offset-6],'poststep'

def cpu_contracts():
    a=np.zeros((2,2,3),np.uint8);b=a.copy();b[0,0,0]=8
    x=pixel_metrics(a,b);assert x['mae']==8/12 and x['max_abs']==8 and x['changed_pixel_fraction']==.25
    assert x['changed_coordinate_fraction']==1/12 and x['squared_error_sum']==64
    mask=np.array([[True,False],[False,False]]);y=pixel_metrics(a,b,mask);assert y['squared_error_sum']==64
    keys=[frame_key(i)for i in range(27)];assert len(set(keys))==27
    assert set(keys)=={(r,c,s)for r in range(3)for c in CASES for s in ['initial','goal','poststep']}
    return dict(pixel_metric_oracle=True,frame_capture_mapping_unique27=True,no_real_environment=True,no_models=True)

def load_parent():
    assert sha(PARENT)==PARENT_SHA
    spec=importlib.util.spec_from_file_location('frozen_cube_reset_v1',PARENT);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);return p

def prepare(args):
    base=B/'runs/cube_reset_state_audit_20261031/plan.json';old=read(base);assert [x['index']for x in old['cases']]==CASES
    site=Path('/efs/prescottsun/project_jepa/envs/lewm/lib/python3.12/site-packages')
    files=[PARENT,site/'ogbench/manipspace/descriptions/floor_wall.xml',site/'mujoco/rendering/classic/renderer.py',
           B/'sources/stable-worldmodel/stable_worldmodel/wrapper/default.py',B/'sources/stable-worldmodel/stable_worldmodel/world/env_pool.py']
    for path,digest in old['source_files'].items():assert sha(path)==digest
    plan=dict(helper_sha256=sha(__file__),parent_helper=str(PARENT),parent_helper_sha256=PARENT_SHA,
        base_plan=str(base),base_plan_sha256=sha(base),source=old['source'],cases=old['cases'],repeats=3,action=old['action'],
        output=args.output,source_files={str(p):sha(p)for p in files},cpu_contracts=cpu_contracts(),
        scope='One same3cases×3freshWorld native reset audit. Exactly9fixed one-step probes, native constructor/random settling retained/counts recorded. Save actual T1 initial/goal/post-step RGB arrays and full camera/light/material/texture/geom visual state plus9post-step segmentation renders for localization only. No learned model, P0, reset correction, texture intervention, new action condition or extension. v1 remains unchanged.',
        frame_scope='AddPixelsWrapper.step renders each step then replaces pixels; EnvPool stores(buf[env,0]) with T=1. Capture asserts shape(1,224,224,3),uint8; no history stack or stale buffer.',
        concrete_visual_source='Native floor_wall.xml skybox uses mark=random,random=.005. V1 omitted texture bytes/material and some visual fields; the new audit measures them without changing them.')
    write(args.plan,plan);print(json.dumps(dict(plan=args.plan,passed=True,cases=CASES)))

def numeric_attributes(obj,prefix,output,depth=0):
    for name in dir(obj):
        if name.startswith('_'):continue
        try:v=getattr(obj,name)
        except (AttributeError,RuntimeError):continue
        key=prefix+'.'+name
        if isinstance(v,np.ndarray):output[key]=v.copy()
        elif isinstance(v,(int,float,bool,np.number)):output[key]=np.asarray(v)
        elif not callable(v) and type(v).__module__.startswith('mujoco') and depth<2:
            numeric_attributes(v,key,output,depth+1)

def visual_state(env,mujoco):
    m=env._model;arrays={}
    for name in dir(m):
        if name.startswith(('cam_','light_','geom_','mat_','tex_','mesh_')):
            value=getattr(m,name)
            if isinstance(value,np.ndarray):arrays['model.'+name]=value.copy()
    numeric_attributes(m.vis,'model.vis',arrays)
    numeric_attributes(env._camera,'env.camera',arrays)
    for name in ['type','fixedcamid','trackbodyid']:arrays['env.camera.'+name]=np.asarray(int(getattr(env._camera,name)),np.int64)
    numeric_attributes(env._scene_option,'env.scene_option',arrays)
    renderer=env._renderer
    if renderer is not None:
        arrays['renderer.scene.flags']=np.asarray(renderer.scene.flags).copy()
        for i,camera in enumerate(renderer.scene.camera):numeric_attributes(camera,f'renderer.camera{i}',arrays)
        for i in range(renderer.scene.nlight):numeric_attributes(renderer.scene.lights[i],f'renderer.light{i}',arrays)
    textures=[]
    data_name='tex_data'if hasattr(m,'tex_data')else'tex_rgb'
    texture_data=np.asarray(getattr(m,data_name))
    for i in range(m.ntex):
        start=int(m.tex_adr[i]);end=int(m.tex_adr[i+1])if i+1<m.ntex else len(texture_data)
        textures.append(dict(id=i,name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_TEXTURE,i),start=start,end=end,
            type=int(m.tex_type[i]),width=int(m.tex_width[i]),height=int(m.tex_height[i]),sha256=array_sha(texture_data[start:end])))
    geoms=[dict(id=i,name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i),
        body_name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])))for i in range(m.ngeom)]
    # This is the expanded scene source, before native compilation's procedural textures.
    from ogbench.manipspace import mjcf_utils
    xml=mjcf_utils.to_string(env._mjcf_model)
    meta=dict(textures=textures,texture_array_key='model.'+data_name,geoms=geoms,
        cube_geom_ids=[int(x)for x in env._cube_geom_ids_list[0]],xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        source_camera='front_pixels',render_shape=[env._render_height,env._render_width])
    return arrays,meta,xml

def run(args):
    plan=read(args.plan);assert plan['helper_sha256']==sha(__file__)and plan['parent_helper_sha256']==PARENT_SHA
    assert sha(plan['base_plan'])==plan['base_plan_sha256']
    for path,d in plan['source_files'].items():assert sha(path)==d
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    write(out/'visual-input-provenance.json',dict(plan=plan,plan_sha256=sha(args.plan),helper_path=str(Path(__file__).resolve()),helper_sha256=sha(__file__)))
    p=load_parent();base=read(plan['base_plan']);base['output']=str(out/'physics')
    native_plan=out/'native-run-plan.json';write(native_plan,base)
    frames={};visuals={};segmentation={};visual_meta={};frame_count=0;state_count=0
    original_hash=p.array_sha;original_snapshot=p.snapshot
    def capture_hash(value):
        nonlocal frame_count
        assert value.shape==(1,224,224,3)and value.dtype==np.uint8,('Not current single-frame RGB',value.shape,value.dtype)
        assert frame_count<27
        frames[frame_key(frame_count)]=value[0].copy();frame_count+=1
        return original_hash(value)
    def capture_snapshot(env,mujoco):
        nonlocal state_count
        state=original_snapshot(env,mujoco)
        # The parent's direct run() call is its post-one-step snapshot; all
        # before/after reset snapshots occur in its named delegate hooks.
        if inspect.currentframe().f_back.f_code.co_name=='run':
            assert state_count<9;key=(state_count//3,CASES[state_count%3]);state_count+=1
            arrays,meta,xml=visual_state(env,mujoco);visuals[key]=arrays;visual_meta[key]=meta
            (out/f'compiled-source-r{key[0]}-case{key[1]}.xml').write_text(xml)
            seg=env.render(camera='front_pixels',segmentation=True)
            assert seg.shape==(224,224,2)and np.issubdtype(seg.dtype,np.integer)
            segmentation[key]=seg.copy();env._renderer.disable_segmentation_rendering()
            after=original_snapshot(env,mujoco)
            assert set(state)==set(after)and all(np.array_equal(v,after[k])for k,v in state.items()),'Segmentation changed recorded physics'
        return state
    p.array_sha=capture_hash;p.snapshot=capture_snapshot
    try:p.run(SimpleNamespace(plan=str(native_plan)))
    finally:p.array_sha=original_hash;p.snapshot=original_snapshot
    assert frame_count==27 and state_count==9
    np.savez_compressed(out/'rgb-and-segmentation.npz',**{f'r{r}_case{c}_{stage}':v for(r,c,stage),v in frames.items()},
        **{f'r{r}_case{c}_segmentation':v for(r,c),v in segmentation.items()})
    np.savez_compressed(out/'visual-model-state.npz',**{f'r{r}_case{c}__{name}':v for(r,c),items in visuals.items()for name,v in items.items()})
    physics=read(out/'physics/results.json');rows=[]
    for case in CASES:
        for a,b in [(0,1),(0,2),(1,2)]:
            first,second=visuals[a,case],visuals[b,case];assert set(first)==set(second)
            changed={k:p.difference(first[k],second[k])for k in first if not np.array_equal(first[k],second[k])}
            seg=segmentation[a,case];meta=visual_meta[a,case]
            robot_ids=[x['id']for x in meta['geoms']if (x['name']or'').startswith('ur5e/')or(x['body_name']or'').startswith('ur5e/')]
            floor_ids=[x['id']for x in meta['geoms']if x['name']=='floor']
            geom=seg[:,:,1]==5 # mjtObj.mjOBJ_GEOM; assert against runtime below.
            import mujoco
            assert int(mujoco.mjtObj.mjOBJ_GEOM)==5
            regions=dict(background=seg[:,:,0]<0,cube=geom&np.isin(seg[:,:,0],meta['cube_geom_ids']),
                         robot=geom&np.isin(seg[:,:,0],robot_ids),floor=geom&np.isin(seg[:,:,0],floor_ids))
            covered=np.logical_or.reduce(list(regions.values()));regions['other']=~covered
            left,right=frames[a,case,'poststep'],frames[b,case,'poststep'];full=pixel_metrics(left,right)
            region_metrics={k:pixel_metrics(left,right,mask)for k,mask in regions.items()}
            for val in region_metrics.values():val['squared_error_fraction']=val['squared_error_sum']/full['squared_error_sum']if full['squared_error_sum']else 0.
            coords=np.argwhere(np.any(left!=right,axis=-1));bbox=None if not len(coords)else[coords.min(0).tolist(),coords.max(0).tolist()]
            pp=next(x for x in physics['comparisons']if x['case_index']==case and x['repeats']==[a,b])
            rows.append(dict(case_index=case,repeats=[a,b],initial=pixel_metrics(frames[a,case,'initial'],frames[b,case,'initial']),
                goal=pixel_metrics(frames[a,case,'goal'],frames[b,case,'goal']),poststep=full,regions=region_metrics,
                difference_bbox_yx=bbox,segmentation_exact=np.array_equal(seg,segmentation[b,case]),
                all_poststep_physical_fields_exact=all(x['exact_equal']for x in pp['stages']['after_one_step'].values()),
                compiled_source_xml_exact=visual_meta[a,case]['xml_sha256']==visual_meta[b,case]['xml_sha256'],
                different_visual_fields=changed,texture_hashes=[visual_meta[a,case]['textures'],visual_meta[b,case]['textures']]))
    result=dict(passed=True,helper_sha256=sha(__file__),parent_helper_sha256=PARENT_SHA,plan_sha256=sha(args.plan),
        frame_shape=[224,224,3],native_time_dimension=1,frames_saved=27,poststep_segmentation_renders=9,
        physics_results=str(out/'physics/results.json'),physics_results_sha256=sha(out/'physics/results.json'),
        native_counts=physics['actual_total_native_counts'],rows=rows,
        visual_metadata={f'r{r}_case{c}':v for(r,c),v in visual_meta.items()},
        artifacts={n:sha(out/n)for n in ['rgb-and-segmentation.npz','visual-model-state.npz']},
        limitations='Exactly the approved same3cases×3freshWorld audit. Segmentation is a descriptive first-repeat region mask and never enters training/control. No texture replacement or reset fix was tested. Covarying texture bytes and background differences do not alone prove the contribution to P0 variation. Preserve the v1 negative one-step physics result.')
    write(out/'results.json',result)
    print(json.dumps(dict(output=str(out),passed=True,rows=[dict(case=x['case_index'],repeats=x['repeats'],poststep=x['poststep'],different_visual_fields=list(x['different_visual_fields']))for x in rows])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','contracts'])
    p.add_argument('--plan',default=str(B/'runs/cube_reset_visual_audit_20261031/plan.json'))
    p.add_argument('--output',default=str(B/'reports/cube-reset-state-20261031/visual-v2'));a=p.parse_args()
    if a.phase=='prepare':prepare(a)
    elif a.phase=='run':run(a)
    else:print(json.dumps(cpu_contracts()))

if __name__=='__main__':main()
