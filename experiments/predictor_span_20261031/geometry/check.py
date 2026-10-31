"""Finite-dimensional geometry witness only; no models, data, labels or GPU."""
import json, math
from pathlib import Path

def chart(v):
 x,y,z=v;return [x,y,z+.3*math.sin(x)*math.cos(y)]
def inverse(v):
 x,y,z=v;return [x,y,z-.3*math.sin(x)*math.cos(y)]
def step_fixed(v,a,b):return [v[0]+a,v[1]+b,v[2]]
def step_chart(v,a,b):return inverse(step_fixed(chart(v),a,b))
def step_field(v,a,b):
 x,y,z=v;return [x+a,y+b,z+(-y*a+x*b)/2]
def loop(step,v,a,b):
 for da,db in [(a,0),(0,b),(-a,0),(0,-b)]:v=step(v,da,db)
 return v
rows=[]
for v,a,b in [([0.,0.,0.],.3,.2),([.4,-.2,.7],.1,-.3),([-.5,.2,-.1],-.2,-.4)]:
 fixed=loop(step_fixed,v,a,b);curved=loop(step_chart,v,a,b);field=loop(step_field,v,a,b)
 fixed_error=max(abs(x-y)for x,y in zip(fixed,v));chart_error=max(abs(x-y)for x,y in zip(curved,v))
 expected=[v[0],v[1],v[2]+a*b];field_error=max(abs(x-y)for x,y in zip(field,expected))
 assert fixed_error<1e-14 and chart_error<1e-14 and field_error<1e-14
 rows.append(dict(initial=v,a=a,b=b,constant_plane_endpoint=fixed,conjugated_plane_endpoint=curved,state_field_endpoint=field,expected_extra_direction=a*b,fixed_error=fixed_error,chart_error=chart_error,field_error=field_error))
result=dict(passed=True,rows=rows,scope='Exact finite four-step toy for general conditional output operators. It is not an implementation or reachable-state proof for the native shared64x16 head; does not imply task improvement or novelty.',model_calls=0,new_data=0,gpu_calls=0)
Path(__file__).with_name('check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,tests=len(rows),max_error=max(max(r[k]for k in ['fixed_error','chart_error','field_error'])for r in rows))))
