"""Compose factual screenshot panels from verified action replays."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for path in ['/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Light.ttc']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    raise RuntimeError('A Chinese font is required for the presentation labels')


def panel(root,task,title,roi,positions,output):
    folder=root/'replay-v2'/f'tf-control-{task}'
    proof=json.loads((folder/'proof.json').read_text())
    n=proof['steps'];success=proof['success']
    frames=[Image.open(folder/'step000.png').convert('RGB'),
            Image.open(folder/f'step{n:03d}.png').convert('RGB'),Image.open(folder/'goal.png').convert('RGB')]
    w=400;gap=16;padding=22;top=112
    crop_h=round(w*(roi[3]-roi[1])/(roi[2]-roi[0]))
    height=top+w+38+crop_h+62
    canvas=Image.new('RGB',(3*w+2*gap+2*padding,height),'#f6f7fb');draw=ImageDraw.Draw(canvas)
    draw.text((padding,12),title,font=font(29),fill='#10223d')
    draw.text((padding,54),'Temporal-forecast · 20 epoch · 训练种子1 · 场景42 · 标准CEM',font=font(18),fill='#42546a')
    headers=['初始画面',f'执行 {n} 步后：'+('成功' if success else '未成功'),'目标参考（缓存专家图像）']
    for i,(im,header) in enumerate(zip(frames,headers)):
        x=padding+i*(w+gap)
        draw.text((x,83),header,font=font(21),fill='#176d45' if i==1 and success else '#10223d')
        overview=im.resize((w,w),Image.Resampling.LANCZOS)
        canvas.paste(overview,(x,top))
        # One identical, explicit image crop across all three columns.
        scaled=tuple(v*w/480 for v in roi)
        draw.rectangle((x+scaled[0],top+scaled[1],x+scaled[2],top+scaled[3]),outline='#ffb321',width=2)
        crop=im.crop(roi).resize((w,crop_h),Image.Resampling.LANCZOS)
        cy=top+w+38;canvas.paste(crop,(x,cy))
        draw.text((x,top+w+6),'同一区域放大',font=font(18),fill='#42546a')
        if positions:
            px,py=positions[i]
            cx=x+(px-roi[0])*w/(roi[2]-roi[0]);yy=cy+(py-roi[1])*crop_h/(roi[3]-roi[1])
            draw.ellipse((cx-20,yy-20,cx+20,yy+20),outline='#ffb321',width=4)
    footer='保存动作逐步重放；成功标记与原评测完全一致。'
    if positions:footer+=f' 橙圈：操作物体；最终距目标 {100*proof["object_final_goal_distance"]:.1f} cm。'
    draw.text((padding,height-43),footer,font=font(18),fill='#42546a')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),task=task,
                source_proof=str(folder/'proof.json'),crop=roi,manually_reviewed_object_pixels=positions,
                transformations='Labeled contact sheet, identical crop, resized images, object circles; scene content unchanged.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    specs=[('bin-picking-v3','Bin picking：夹爪移到蓝盒，绿色方块仍在红盒',
            (205,205,410,370),[(283,263),(283,263),(324,298)],'bin-picking.png'),
           ('pick-place-v3','Pick & place：夹爪移动了，红色物体仍留在桌上',
            (200,130,430,355),[(299,293),(299,293),(325,208)],'pick-place.png'),
           ('reach-v3','Reach 成功对照：夹爪在40步内到达目标附近',
            (200,130,430,355),None,'reach-success.png')]
    records=[panel(root,task,title,roi,positions,root/name) for task,title,roi,positions,name in specs]
    (root/'PANELS.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
