"""Standard-library CPU witness; no files/data/models beyond its own JSON output."""
import itertools,json,math,random
from pathlib import Path


def parity(x):return 1 if bin(x).count('1')%2==0 else -1

def code(x,rows):return [parity(x&r)for r in rows]

def rank(rows,width):
    a=list(rows);k=0
    for j in range(width):
        pivot=next((i for i in range(k,len(a))if a[i]&(1<<j)),None)
        if pivot is None:continue
        a[k],a[pivot]=a[pivot],a[k]
        for i in range(len(a)):
            if i!=k and a[i]&(1<<j):a[i]^=a[k]
        k+=1
    return k


def main():
    # D=9 raw bits: eight identical background bits and one independent small signal.
    repeat=8;d=repeat+1;rows=list(range(1<<d));values={}
    for b,s in itertools.product([0,1],repeat=2):
        x=((1<<repeat)-1)*b+(s<<repeat);values[(b,s)]=code(x,rows)
    risks={}
    for retained in [0,1]:
        error=0.
        for bs,c in values.items():
            group=[other for key,other in values.items()if key[retained]==bs[retained]]
            mean=[sum(v)/len(v)for v in zip(*group)]
            error+=sum((a-b)**2 for a,b in zip(c,mean))/len(c)/4
        risks[['background','small_signal'][retained]]=error
    assert risks==dict(background=.5,small_signal=.5)
    # The full parity feature kernel is exactly delta(x,y), with all nonidentical
    # source strings equally distant regardless of changed area.
    states=list(values);kernel=[]
    for x in states:
        line=[]
        for y in states:
            k=sum(a*b for a,b in zip(values[x],values[y]))/len(rows)
            assert k==int(x==y);line.append(k)
        kernel.append(line)
    # Small-change amplification + lossless observation do not guarantee prediction.
    # Coordinates: N0,B,S,N1,...,N7. Every checksum includes unpredictable N0.
    width=10;dim=64;rng=random.Random(20261031)
    noisy_rows=[1]+[1|(1<<j)for j in range(1,width)]
    noisy_rows += [1|(rng.getrandbits(width-1)<<1)for _ in range(dim-width)]
    assert rank(noisy_rows,width)==width
    allcodes={tuple(code(x,noisy_rows))for x in range(1<<width)};assert len(allcodes)==1<<width
    means={}
    for b,s in itertools.product([0,1],repeat=2):
        observations=[]
        for n in range(1<<8):
            x=(n&1)|(b<<1)|(s<<2)|((n>>1)<<3)
            observations.append(code(x,noisy_rows))
        means[(b,s)]=[sum(v)/len(v)for v in zip(*observations)]
        assert means[(b,s)]==[0]*dim
    goal=code(0,noisy_rows)
    forecast_costs=[sum((v-g)**2 for v,g in zip(means[(0,s)],goal))/dim for s in [0,1]]
    assert forecast_costs==[1.,1.]
    raw_costs=[]
    for s in [0,1]:
        raw=[0.,1.,parity(s)]+[0.]*7
        raw_costs.append(sum((v-1.)**2 for v in raw)/width)
    assert raw_costs==[.8,1.2]
    flip_fraction=sum(bool(r&(1<<2))for r in noisy_rows)/dim
    result=dict(passed=True,seed=20261031,calls=dict(model=0,gpu=0,data=0),
      area_example=dict(background_copies=repeat,raw_mse_keep_background=1/(repeat+1),raw_mse_keep_small_signal=repeat/(repeat+1),checksum_risks=risks),
      kernel=dict(full_parity_kernel=kernel,all_distinct_normalized_squared_distance=2),
      counterexample=dict(width=width,code_dimension=dim,rank=rank(noisy_rows,width),distinct_observations=len(allcodes),
        distinct_codes=len(allcodes),object_flip_fraction=flip_fraction,all_conditional_mean_codes_zero=True,
        checksum_goal_cost_correct_wrong=forecast_costs,raw_goal_cost_correct_wrong=raw_costs,
        object_observably_decodable='S recovered exactly from parity row0 times parity row2; encoding is injective'),
      iid_hash_object_signal_survival=[dict(unpredictable_bits=m,single_row_probability=2**(-m-1),
        probability_at_least_one_of64=1-(1-2**(-m-1))**64)for m in [0,2,4,8,16]])
    Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
