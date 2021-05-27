import os
def do_job(mode):
    if not os.path.exists(mode):
        os.mkdir(mode)
    f=open('scannet_'+mode+'.txt')
    if mode == 'test':
        scan_folder='scans_test'
    else:
        scan_folder='scans'
    scene_ids=f.read().splitlines()
    for scene_id in scene_ids:
        files_in_scene=os.listdir('scannet/'+scan_folder+'/'+scene_id)
        for file_in_scene in files_in_scene:
            os.rename('scannet/'+scan_folder+'/'+scene_id+'/'+file_in_scene,mode+'/'+file_in_scene)


do_job('train')
#do_job('test')
do_job('val')
